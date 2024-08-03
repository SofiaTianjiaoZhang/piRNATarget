import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embedding_dim, pad=False, device='cpu'):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.pad = pad
        self.seq_len=seq_len
        self.device=device

    def get_embedding(seq_len, embedding_dim, padding_idx=None):
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

    def forward(self, x, seq_lens=[]):
        if self.pad:
            for idx, i in enumerate(x):
                pos_embedding = SinusoidalPositionalEmbedding.get_embedding(self.seq_len, self.embedding_dim,
                                                                            padding_idx=list(range(seq_lens[idx], self.seq_len)))
                pos_embedding = torch.cat([pos_embedding, pos_embedding], dim=0).to(self.device)
                x[idx] = i + pos_embedding
        else:
            pos_embedding = SinusoidalPositionalEmbedding.get_embedding(self.seq_len, self.embedding_dim,
                                                                        padding_idx=None)
            pos_embedding = torch.cat([pos_embedding, pos_embedding], dim=0).to(self.device)
            x = x + pos_embedding

        return x



class SEblock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels//reduction)
        self.relu = nn.PReLU()#(inplace=True)
        self.fc2 = nn.Linear(channels//reduction, channels)
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)

        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        input_x = x
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1)     #B, CNN_dim, 1   the importance of every channel
        return input_x * x



class FELayer(nn.Module):
    def __init__(self, layer_infos, last_norm=True, norm_type='batch', bias=True):
        super(FELayer, self).__init__()
        self.linear_layers = nn.Sequential()
        for idx, li in enumerate(layer_infos):
            self.linear_layers.add_module(f'linear_{idx}', nn.Linear(li[0], li[1], bias=bias))
            if idx != len(layer_infos) - 1 or (idx == len(layer_infos) - 1 and last_norm):
                self.linear_layers.add_module(f'bn_{idx}',
                                              nn.LayerNorm(li[1]) if norm_type != 'batch' else nn.BatchNorm1d(li[1]))
                self.linear_layers.add_module(f'relu_{idx}', nn.PReLU())
                if len(li) == 3:
                    self.linear_layers.add_module(f'dropout_{idx}', nn.Dropout(li[2]))

    def forward(self, x):
        return self.linear_layers(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reduction=4, add_residual=False, res_dim=16):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=(kernel_size // 2))

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.PReLU()
        self.se = SEblock(channels=out_channels, reduction=reduction)

        if add_residual:
            self.conv2 = nn.Conv1d(in_channels=res_dim, out_channels=out_channels, kernel_size=1)
            self.bn2 = nn.BatchNorm1d(out_channels)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x, residual=None):
        x = x.permute(0, 2, 1)
        x = self.conv(x)   # conv input is B * E * L  conv output is B * E(out) * L
        x = self.bn1(x)
        if residual is not None:
            x = x + self.bn2(self.conv2(residual))   #batchNorm output B * E(out) * L
        x = self.relu(x)
        x = self.se(x)
        #output is B * E(out) * L
        return x

class PositionWiseFFN(nn.Module):
    def __init__(self, model_dim, dropout=0.0):
        super(PositionWiseFFN, self).__init__()
        dff = model_dim * 4
        self.l = nn.Linear(model_dim, dff)
        self.o = nn.Linear(dff, model_dim)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        o = self.relu(self.l(x))
        o = self.o(o)
        o = self.dropout(o)

        o = self.layer_norm(x + o)

        return o  # [n, len, dim]


class ConvAtt(nn.Module):
    def __init__(self, emb_dim=4, cnn_dim=64, kernel=7, se_reduction=4, cnn_dropout=0.3, mrna_len=35, pirna_len=35,
                 nhead=4, att_dropout=0., device='cpu'):
        super(ConvAtt, self).__init__()
        self.device=device
        self.pirna_len = pirna_len
        self.mrna_len = mrna_len

        self.pirna_conv = ConvBlock(emb_dim, cnn_dim, kernel, stride=1,
                                    reduction=se_reduction)  # output is B * cnn_dim * L
        self.pirna_dropout = nn.Dropout(cnn_dropout)

        self.mrna_conv = ConvBlock(emb_dim, cnn_dim, kernel, stride=1, reduction=se_reduction)
        self.mrna_dropout = nn.Dropout(cnn_dropout)

        self.d_model = cnn_dim
        self.multihead_attn = nn.MultiheadAttention(self.d_model, nhead, dropout=att_dropout)
        self.pos_embed = SinusoidalPositionalEmbedding(seq_len=self.pirna_len, embedding_dim=self.d_model, pad=False, device=self.device)

    def generate_mask(self, seq):
        """input is pirna_x or mrna_x (tensor) B * L * E, assume padding char is 0 tensor"""
        #check zeros seq.sum(dim=2)
        mask_matrix = torch.Tensor(seq.sum(dim=2)) == 0
        return mask_matrix

    def forward(self, x):
        pirna_x = x[:, :self.pirna_len, :]
        mrna_x = x[:, self.mrna_len:, :]  # B * L * E
        mask_matrix = self.generate_mask(pirna_x)
        ##cnn embedding
        pirna_value = self.pirna_dropout(self.pirna_conv(pirna_x))
        mrna_value = self.mrna_dropout(self.mrna_conv(mrna_x))   #B * E(cnn out) * L
        

        ##add position embedding here
        pirna = pirna_value.permute(0, 2, 1)
        mrna = mrna_value.permute(0, 2, 1)
        all = torch.cat([pirna, mrna], dim=1)
        all_pos_embed = self.pos_embed(all)
        #经验证这里不需 positional embedding
        pirna_value = all[:, :self.pirna_len, :].permute(1, 0, 2)
        mrna_value = all[:, self.mrna_len:, :].permute(1, 0, 2)

        ##att
        #multihead_attn needs L * B * E
        att, att_score = self.multihead_attn(pirna_value, mrna_value, mrna_value, key_padding_mask=mask_matrix)
        # att L B E       att_score L L
        return att, att_score, mask_matrix


class ConvHierAttention3(nn.Module):
    def __init__(self, emb_dim=4, cnn_dim=64, se_reduction=4, cnn_dropout=0., mrna_len=35, pirna_len=35, nhead=4,
                 att_dropout=0., cls_dropout=0., fnn_dropout=0., device='cpu'):
        super(ConvHierAttention3, self).__init__()
        self.device=device
        self.cov_att = ConvAtt(emb_dim=emb_dim, cnn_dim=cnn_dim, kernel=5, se_reduction=se_reduction,
                               cnn_dropout=cnn_dropout, mrna_len=mrna_len, pirna_len=pirna_len, nhead=nhead,
                               att_dropout=att_dropout, device=self.device)
        self.norm_att = nn.LayerNorm(cnn_dim)
        self.att1_ffn = PositionWiseFFN(cnn_dim, fnn_dropout)
        ##SecondAttention
        self.d_model = cnn_dim
        self.secatt = nn.MultiheadAttention(self.d_model, nhead, dropout=fnn_dropout)
        self.att2_ffn = PositionWiseFFN(cnn_dim, fnn_dropout)

        # Classification
        self.cls_dropout_layer = nn.Dropout(cls_dropout)
        self.cls_input_dim = mrna_len * cnn_dim
        self.cls_layer = FELayer([
            [self.cls_input_dim, self.cls_input_dim // 4, cls_dropout],
            [self.cls_input_dim // 4, self.cls_input_dim // 16, cls_dropout],
            [self.cls_input_dim // 16, 2]
        ], last_norm=False, norm_type='batch')

    def forward(self, x):
        att, att_score, mask_matrix = self.cov_att(x)
        att2, att2_score = self.secatt(att, att, att, key_padding_mask=mask_matrix)
        att = att.permute(1, 0, 2)
        att = self.norm_att(att)
        att = self.att1_ffn(att)
        att2 = att2.permute(1, 0, 2)
        att2 = self.norm_att(att2)
        att2 = self.att2_ffn(att2)
        out = att + att2
        out = self.norm_att(out)
        out = self.cls_dropout_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.cls_layer(out)

        return out, att_score, att2_score
