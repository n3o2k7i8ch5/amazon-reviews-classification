import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_loader import load_data, save_data, ALL_POS_TAGS, create_data
from gpu_stuff import get_gpu_memory_map
from lstm_models import PosLSTM
from super_tensor_converter import to_super_tensor_from_list, to_super_tensor_from_tensor

train_sentences = None
train_pos = None
train_labels = None
test_sentences = None
test_pos = None
test_labels = None
word2idx = None

try:
    print('Loading data...')
    train_sentences = load_data('data/train_sentences.data')
    train_pos = load_data('data/train_pos.data')
    train_labels = load_data('data/train_labels.data')
    test_sentences = load_data('data/test_sentences.data')
    test_pos = load_data('data/test_pos.data')
    test_labels = load_data('data/test_labels.data')
    word2idx = load_data('data/word2idx.data')
    print('Data loaded.')
except FileNotFoundError:
    print('Creating data...')
    train_sentences, train_pos, train_labels, test_sentences, test_pos, test_labels, word2idx = create_data()
    save_data(train_sentences, 'data/train_sentences.data')
    save_data(train_pos, 'data/train_pos.data')
    save_data(train_labels, 'data/train_labels.data')
    save_data(test_sentences, 'data/test_sentences.data')
    save_data(test_pos, 'data/test_pos.data')
    save_data(test_labels, 'data/test_labels.data')
    save_data(word2idx, 'data/word2idx.data')
    print('Data created.')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

try:
    print('Loading super-tensors')
    test_super_tensors = load_data('test_super_tensors.data')
    print('Super-tensors loaded')
except FileNotFoundError:
    print('Creating super-tensors')
    test_super_tensors = to_super_tensor_from_list(test_sentences, test_pos)
    save_data(test_super_tensors, 'test_super_tensors.data')

INPUT_SIZE = 200  # features
HIDDEN_SIZE = 132

EPOCHS = 2
BATCH_SIZE = 128
BATCH_SIZE_TEST = 64 * 64 * 2

MODEL_DUMP_PATH = f'./pos_state_dict_{INPUT_SIZE}_{HIDDEN_SIZE}.pt'


def print_metrics(test_loader: DataLoader, epoch: int, n_batch: int, criterion):
    with torch.no_grad():

        model.eval()
        torch.no_grad()

        val_h = None
        eval_counter: int = 0

        val_losses = []
        bad_hits = 0
        total_hits = 0

        for super_inp, lab, in test_loader:

            eval_counter += 1
            if eval_counter % 2 != 0:
                continue

            sys.stdout.write(
                f'\rEpoch {epoch}, progress: {n_batch / len(train_loader) * 100} - counting loss: {i / len(test_loader) * 100}%'
            )
            sys.stdout.flush()

            if val_h is not None:
                val_h = tuple([each.data for each in val_h])

            super_inp, lab = super_inp.to(device), lab.to(device)

            if len(super_inp) != BATCH_SIZE_TEST:
                continue

            out, val_h = model(super_inp, val_h)
            loss = criterion(out.squeeze(), lab.squeeze().float())
            val_losses.append(loss.item())

            out = (out + 0.5).int().squeeze().clamp(0, 1)

            bad_hits += torch.abs(out - lab).sum().item()
            total_hits += BATCH_SIZE_TEST

        print()
        print("Epoch: {}/{}...".format(epoch + 1, EPOCHS),
              "Step: {}...".format(n_batch),
              "Loss: {:.6f}...".format(loss.detach().item()),
              "Valid Loss: {:.6f}".format(np.mean(val_losses)),
              "Accuracy: " + str(1 - bad_hits / total_hits),
              'GPU mem: ' + str(get_gpu_memory_map()))

    model.train()
    torch.enable_grad()


model = PosLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    out_size=1,
    vocab_size=len(word2idx) + 1,
    pos_count=len(ALL_POS_TAGS),
).to(device)

train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels),torch.from_numpy(train_pos))
del train_sentences, train_labels, train_pos

test_data = TensorDataset(torch.from_numpy(np.array(test_super_tensors)), torch.from_numpy(test_labels))
del test_super_tensors, test_labels

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE_TEST)

del train_data, test_data

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if os.path.exists(MODEL_DUMP_PATH):
    model.load_state_dict(torch.load(MODEL_DUMP_PATH))
    print('Model state restored')
    print_metrics(test_loader, -1, -1, criterion)

model.train()
torch.enable_grad()
for epoch in range(EPOCHS):
    hidden = None

    for n_batch, (inputs, labels, pos) in enumerate(train_loader):

        sys.stdout.write(
            f'\rEpoch {epoch}, progress: {(n_batch + 1) / len(train_loader) * 100}, gpu mem: {get_gpu_memory_map()})'
        )
        sys.stdout.flush()

        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()

        if hidden is not None:
            hidden = (hidden[0].detach(), hidden[1].detach())

        super_input_batch = []
        for i in range(len(inputs)):
            super_input_batch.append(to_super_tensor_from_tensor(inputs[i], pos[i]))

        super_input_batch = torch.tensor(super_input_batch, device=device)
        output, hidden = model(super_input_batch.detach(), hidden)

        loss = criterion(output.squeeze(), labels.squeeze().float())
        loss.backward(retain_graph=True)
        optimizer.step()

        torch.cuda.empty_cache()

        if n_batch % 100 == 0:
            print_metrics(test_loader, epoch, n_batch, criterion)

            print('Saving model...')
            torch.save(model.state_dict(), MODEL_DUMP_PATH)
