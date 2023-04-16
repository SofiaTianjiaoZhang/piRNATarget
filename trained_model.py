import torch
from pre_model import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0., bidirectional=False):
        super(MaskedLSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=bias,
             batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_tensor):
        # input_tensor shape: batch_size*time_step*dim , seq_lens: (batch_size,)  when batch_first = True
        total_length = input_tensor.size(1) if self.batch_first else input_tensor.size(0)
        seq_lens = get_seq_len(input_tensor, batch_first = self.batch_first)
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(input_tensor, seq_lens, batch_first=self.batch_first, enforce_sorted=False)
        y_lstm, hidden = self.lstm(x_packed)
        y_padded, length = torch.nn.utils.rnn.pad_packed_sequence(y_lstm, batch_first=self.batch_first, total_length=total_length)
        return y_padded, hidden


class BiLSTM_Attention(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_layers, mask=True, dropout=0., pirna_max_len = 34):
        super(BiLSTM_Attention, self).__init__()
        self.dropout = dropout
        self.rnn = MaskedLSTM(input_size=embedding_dim,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               batch_first=False,
                               bidirectional=True,
                               dropout=self.dropout)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(
            num_hiddens * 2, num_hiddens * 2))
        self.u_omega = nn.Parameter(torch.Tensor(num_hiddens * 2, 1))
        self.decoder = nn.Linear(2 * num_hiddens, 2)
        self.mask = mask
        self.pirna_max_len = pirna_max_len
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        inputs = inputs.permute(1,0,2)
        outputs, (h, c) = self.rnn(inputs)

        x = outputs.permute(1, 0, 2)
        #x (batch, seq_len, 2 * num_hiddens)
        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)
        if self.mask:
            seq_lens = get_seq_len(inputs, batch_first=False)
            global piRNA_max_len
            mask_matrix = torch.Tensor([[1] * i + [0] * (self.pirna_max_len-i) for i in seq_lens])
            mask_matrix = mask_matrix.unsqueeze(-1)
            att = att.masked_fill(mask_matrix == 0,-1e9)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束

        feat = torch.sum(scored_x, dim=1)  # 加权求和
        # feat形状是(batch_size, 2 * num_hiddens)
        outs = self.decoder(feat)
        # out形状是(batch_size, 2)
        return outs, att_score