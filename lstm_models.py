from typing import Tuple, Optional

import torch
from torch import nn

from enum import IntEnum

from torch.nn.parameter import Parameter


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class PosLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, out_size: int, vocab_size: int, pos_count: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.pos_count = pos_count

        self.embedding = nn.Embedding(vocab_size, input_size, padding_idx=0)

        # forget gate
        self.W_forget_x = Parameter(torch.Tensor(pos_count, input_size, hidden_size))
        nn.init.xavier_uniform_(self.W_forget_x)
        self.W_forget_h = Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.W_forget_h)
        self.b_forget = Parameter(torch.Tensor(hidden_size))
        nn.init.zeros_(self.b_ponadforget)

        # input gate
        self.W_input_contr_x = Parameter(torch.Tensor(pos_count, input_size, hidden_size))
        nn.init.xavier_uniform_(self.W_input_contr_x)
        self.W_input_contr_h = Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.W_input_contr_h)
        self.b_input_contr = Parameter(torch.Tensor(hidden_size))
        nn.init.zeros_(self.b_input_contr)

        # ???
        self.W_input_x = Parameter(torch.Tensor(pos_count, input_size, hidden_size))
        nn.init.xavier_uniform_(self.W_input_x)
        self.W_input_h = Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.W_input_h)
        self.b_input = Parameter(torch.Tensor(hidden_size))
        nn.init.zeros_(self.b_input)

        # output gate
        self.W_out_x = Parameter(torch.Tensor(pos_count, input_size, hidden_size))
        nn.init.xavier_uniform_(self.W_out_x)
        self.W_out_h = Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.W_out_h)
        self.b_out = Parameter(torch.Tensor(hidden_size))
        nn.init.zeros_(self.b_out)

        self.layer_out = nn.Linear(hidden_size, out_size, bias=True)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(
        self,
        x: torch.Tensor,
        init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # x is of shape (batch, sequence, feature). Features will be created in the embedding layer.
        bs, pos_count, seq_sz = x.size()
        x = x.permute(1, 0, 2)

        if init_states is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):  # iterate over the time steps
            x_t = x[:, :, t]
            x_t = self.embedding(x_t)

            f_t = torch.sigmoid(
                torch.sum(x_t @ self.W_forget_x, dim=0) +
                h_t @ self.W_forget_h +
                self.b_forget)

            i_t = torch.sigmoid(
                torch.sum(x_t @ self.W_input_contr_x, dim=0) +
                h_t @ self.W_input_contr_h +
                self.b_input_contr)

            g_t = torch.tanh(
                torch.sum(x_t @ self.W_input_x, dim=0) +
                h_t @ self.W_input_h +
                self.b_input)

            o_t = torch.sigmoid(
                torch.sum(x_t @ self.W_out_x, dim=0) +
                h_t @ self.W_out_h +
                self.b_out)

            inp = i_t * g_t

            c_t = f_t * c_t + inp
            h_t = o_t * torch.tanh(c_t)

        out = self.layer_out(h_t)

        return out, (h_t, c_t)


class OptimizedLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, out_size: int, vocab_size: int, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, input_size, padding_idx=0)

        self.layer_ih = nn.Linear(input_size, hidden_size * 4, bias=True)
        self.layer_hh = nn.Linear(hidden_size, hidden_size * 4, bias=True)
        self.layer_out = nn.Linear(hidden_size, out_size, bias=True)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor,
                init_states: Optional[Tuple[torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""

        bs, seq_sz = x.size()

        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size, device=x.device, requires_grad=True),
                        torch.zeros(self.hidden_size, device=x.device, requires_grad=True))
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):

            x_t = x[:, t]
            x_t = self.embedding(x_t)

            # batch the computations into a single matrix multiplication
            gates = self.layer_ih(x_t) + self.layer_hh(h_t)

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :self.hidden_size]),  # input
                torch.sigmoid(gates[:, self.hidden_size:2*self.hidden_size]),  # forget
                torch.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size]),
                torch.sigmoid(gates[:, 3*self.hidden_size:]),  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

        out = self.layer_out(h_t)

        return out, (h_t, c_t)
