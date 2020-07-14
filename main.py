import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import data_loader
from gpu_stuff import get_gpu_memory_map
from my_lstm import PosLSTM, OptimizedLSTM

train_sentences = None
train_labels = None
test_sentences = None
test_labels = None
word2idx = None

try:
    print('Loading data...')
    train_sentences = data_loader.load_data('train_sentences.data')
    train_labels = data_loader.load_data('train_labels.data')
    test_sentences = data_loader.load_data('test_sentences.data')
    test_labels = data_loader.load_data('test_labels.data')
    word2idx = data_loader.load_data('word2idx.data')
    print('Data loaded.')
except IOError:
    print('Creating data...')
    train_sentences, train_pos, train_labels, test_sentences, test_pos, test_labels, word2idx = data_loader.create_data()
    data_loader.save_data(train_sentences, 'train_sentences.data')
    data_loader.save_data(train_pos, 'train_pos.data')
    data_loader.save_data(train_labels, 'train_labels.data')
    data_loader.save_data(test_sentences, 'test_sentences.data')
    data_loader.save_data(test_pos, 'test_pos.data')
    data_loader.save_data(test_labels, 'test_labels.data')
    data_loader.save_data(word2idx, 'word2idx.data')
    print('Data created.')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

input_size = 200
hidden_size = 132

epochs = 2
batch_size = 32
batch_size_test = 64 * 64 * 2

file_name = './state_dict_' + str(input_size) + '_' + str(hidden_size) + '.pt'


def printMetrics(test_loader, epoch, counter, criterion):
    with torch.no_grad():

        model.eval()
        torch.no_grad()

        val_h = None
        eval_counter: int = 0

        val_losses = []
        bad_hits = 0
        total_hits = 0

        for inp, lab in test_loader:

            eval_counter += 1

            sys.stdout.write('\rEpoch ' + str(epoch) + ', progress: ' + str(
                counter / len(train_loader) * 100) + ' - counting loss: ' + str(
                eval_counter / len(test_loader) * 100) + '%')
            sys.stdout.flush()

            if val_h is not None:
                val_h = tuple([each.data for each in val_h])

            inp, lab = inp.to(device), lab.to(device)

            if len(inp) != batch_size_test:
                continue

            out, val_h = model(inp, val_h)
            loss = criterion(out.squeeze(), lab.float())
            val_losses.append(loss.item())

            out = (out + 0.5).int().squeeze().clamp(0, 1)
            # _, out = torch.min(out, 1)
            # _, out = torch.max(out, 0)

            bad_hits += torch.abs(out - lab).sum().item()
            total_hits += batch_size_test

        print()
        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Step: {}...".format(counter),
              # "Loss: {:.6f}...".format(loss.detach().item()),
              "Valid Loss: {:.6f}".format(np.mean(val_losses)),
              "Accuracy: " + str(1 - bad_hits / total_hits),
              'GPU mem: ' + str(get_gpu_memory_map()))

    model.train()
    torch.enable_grad()


model = OptimizedLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    out_size=1,
    vocab_size=len(word2idx) + 1,
    device=device
).to(device)

train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size_test)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

if os.path.exists(file_name):
    model.load_state_dict(torch.load(file_name))
    print('Model state restored')
    printMetrics(test_loader, -1, -1, criterion)

model.train()
torch.enable_grad()
for epoch in range(epochs):
    h = None
    # h = model.init_hidden(batch_size)

    counter = 0
    for inputs, labels in train_loader:
        counter += 1
        # print(counter)

        sys.stdout.write('\rEpoch ' + str(epoch) + ', progress: ' + str(counter / len(train_loader) * 100) + ', gpu mem: ' + str(get_gpu_memory_map()))
        sys.stdout.flush()

        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()

        if h is not None:
            h = (h[0].detach(), h[1].detach()) # .detatch() sprawia, że nie ma memory leaków

        output, h = model(inputs.detach(), h)

        loss = criterion(output.squeeze(), labels.float())
        loss.backward(retain_graph=True)  # keep_greph z jakiegoś powodu nie działa (?)
        optimizer.step()
        #sys.stdout.write("\rLoss: {:.6f}...".format(loss.item()))
        # sys.stdout.flush()

        torch.cuda.empty_cache()

        if counter % 20 == 0:
            printMetrics(test_loader, epoch, counter, criterion)

            print('Saving model...')
            torch.save(model.state_dict(), file_name)
