import torch
import pandas as pd
import math
import torch.nn as nn


def read_from_csv(file = "/Users/zhangtianjiao/work/piRNA_target/predict/input/Demo.csv"):
    data_frame = pd.read_csv(file)
    return data_frame


def data_preprocessing(data_frame, device='cpu', have_ans=True, pirna_max_len=35):
    onehot = {ele:([0]*idx + [1] + [0]*(3-idx)) for idx, ele in enumerate(['A', 'T', 'C', 'G'])}
    x1 = []
    x2 = []
    for i in range(0, data_frame.shape[0]):
        pirna = data_frame.iloc[i, 0]
        mrna = data_frame.iloc[i, 1]
        padding_matrix = [[0] * len(onehot['A']) for i in range(pirna_max_len - len(pirna))]
        x1.append([onehot[c] for c in pirna] + padding_matrix)
        x2.append([onehot[c] for c in mrna] + padding_matrix)
    tensor_x1 = torch.tensor(x1).float()
    tensor_x2 = torch.tensor(x2).float()
    tensor_x = torch.cat((tensor_x1, tensor_x2), 1).to(device=device)
    if have_ans:
        y = [i for i in data_frame['Label']]
        tensor_y = torch.tensor(y).int().to(device=device)
        return tensor_x, tensor_y
    else:
        return tensor_x

def get_seq_len(inputs, batch_first=True):
    """
    :param inputs: three-dimension torch.ternsor (eg. batch, seq, feature)
    :param batch_first: if batch is first
    :return: a list of sequence length
    """
    len_list = []
    inputs = inputs.cpu()
    data = inputs if batch_first is True else inputs.permute(1, 0, 2)
    for i in data:
        ind = [not x.equal(torch.Tensor([0.] * inputs.shape[-1])) for x in i]
        len_list.append(len(i[ind]))
    return len_list


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embedding_dim, pad = True):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.pad = pad
        
    def get_embedding(seq_len, embedding_dim, padding_idx = None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(seq_len, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(seq_len, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(seq_len, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb
    
    def get_seq_len(inputs, batch_first=True):
        """
        :param inputs: three-dimension torch.ternsor (eg. batch, seq, feature)
        :param batch_first: if batch is first
        :return: a list of sequence length
        """
        len_list = []
        inputs = inputs.cpu()
        data = inputs if batch_first is True else inputs.permute(1, 0, 2)
        for i in data:
            ind = [not x.equal(torch.Tensor([0.] * inputs.shape[-1])) for x in i]
            len_list.append(len(i[ind]))
        return len_list
    
    def forward(self, x):
        if self.pad:
            seq_lens = SinusoidalPositionalEmbedding.get_seq_len(x[:, :self.seq_len,:])
            for idx, i  in enumerate(x):
                pos_embedding = SinusoidalPositionalEmbedding.get_embedding(self.seq_len, self.embedding_dim, padding_idx = list(range(seq_lens[idx], self.seq_len)))
                pos_embedding = torch.cat([pos_embedding, pos_embedding], dim=0)
                x[idx] = i + pos_embedding
        else:
            pos_embedding = get_embedding(self.seq_len, self.embedding_dim, padding_idx = None)
            pos_embedding = torch.cat([pos_embedding, pos_embedding], dim=0)
            x = x + pos_embedding
        
        return x
        
    
        
