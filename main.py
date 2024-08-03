import sys
import os
import pandas as pd
from pre_model1 import *
from model_final import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


num_hiddens = 64
nhead = 4
att_dropout = 0.1
cls_dropout = 0.3
label_smoothing = 0.1
cnn_dropout = 0.3
fnn_dropout = 0.5
avail_device='cpu'


def get_model(load_model):
    model = ConvHierAttention3(emb_dim=4, cnn_dim=num_hiddens, se_reduction=4, cnn_dropout=cnn_dropout, mrna_len=35,
                               pirna_len=35, nhead=nhead, att_dropout=att_dropout, fnn_dropout=fnn_dropout,
                               cls_dropout=cls_dropout, device=avail_device).to(avail_device)
    model.load_state_dict(state_dict=load_model.state_dict(), strict=True)
    return model


def prediction(model, seq, thres = 0.5):
    model.eval()
    with torch.no_grad():
        for data in seq:
            data = torch.unsqueeze(data, dim=0)
            output = model(data)[0]
            curr_pred = 1 if F.softmax(output, dim=-1)[:, 1] > thres else 0
            prediction = ['NonTarget','Target'][curr_pred]
            print(prediction)



if '__main__' == __name__:
    '''
        *** USAGE ***
        python main.py --input_file_name test_dataset.csv
    '''
    parser = argparse.ArgumentParser(description='piRNA targeting prediction from csv file')
    parser.add_argument('--input_file_name', type=str)
    parser.add_argument('--thres', type=float)
    opt = parser.parse_args()
    thres=opt.thres
    load_model = torch.load('final_model.pth')
    model = get_model(load_model=load_model)
    input_data = read_from_csv(file=opt.input_file_name)
    piRNA_max_len = max(input_data.iloc[:, 1].apply(len, 1))
    seq = data_preprocessing(data_frame=input_data, have_ans=False, pirna_max_len=piRNA_max_len)
    prediction(model=model, seq=seq, thres=thres)