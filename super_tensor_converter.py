import sys

import torch

import data_loader


# intakes a 1D vector, only from from a batch
# outputs a 2D tensor
def to_super_tensor_from_tensor(input: torch.Tensor, pos) -> list:
    super_tensor = []
    for tag_index in range(len(data_loader.all_pos_tags)):
        merged_vals = []
        for i in range(input.size()[0]):
            merged_vals.append(input[i].item() if pos[i] == tag_index else 0)

        super_tensor.append(merged_vals)

    return super_tensor


def to_super_tensor_from_list(input: list, pos: list) -> list:
    super_tensors = []

    for i in range(len(input)):

        if i % 2000 == 0:
            sys.stdout.write('\rCreating test super-tensor... ' + str(i / len(input) * 100) + '%\n')
            sys.stdout.flush()

        sup_tens = []
        for tag_index in range(len(data_loader.all_pos_tags)):

            row = []
            for j in range(len(input[0])):
                row.append(input[i][j].item() if pos[i][j] == tag_index else 0)

            sup_tens.append(row)

        super_tensors.append(sup_tens)

    return super_tensors
