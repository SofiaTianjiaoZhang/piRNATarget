from pre_model import *
import argparse
from trained_model import *
import torch
import torch.nn as nn
import torch.nn.functional as F

thres = 0.5

def get_model(load_model):
    model = BiLSTM_Attention(embedding_dim=3, num_hiddens=4, num_layers=2, mask=True, dropout=0.,
                             pirna_max_len=piRNA_max_len)
    model.load_state_dict(state_dict=load_model.state_dict(), strict=True)
    return model


def prediction(model, seq):
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
    opt = parser.parse_args()
    input_data = read_from_csv(file=opt.input_file_name)
    piRNA_max_len = max(input_data.iloc[:, 1].apply(len, 1))
    seq = data_preprocessing(data_frame=input_data, have_ans=False, pirna_max_len=piRNA_max_len)
    load_model = torch.load('trained_model.pth')
    model = get_model(load_model=load_model)
    prediction(model=model, seq=seq)
