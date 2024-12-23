import sys
import os
#sys.path.append('/Users/zhangtianjiao/workplace/piRNATarget_all/draft/revise/fine-tuning/upload')
import pandas as pd
from pre_model1 import *
from model_final import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pysam
import re


num_hiddens = 64
nhead = 4
att_dropout = 0.1
cls_dropout = 0.3
label_smoothing = 0.1
cnn_dropout = 0.3
fnn_dropout = 0.5
avail_device='cpu'


def verify(sequence):
    '''This code verfies if a sequence is a DNA or RNA'''
    # set the input sequence
    sequence = sequence.upper()
    seq = set(sequence)

    # confirm if its elements is equal to the
    # set of valid DNA bases
    # Use a union method to ensure the sequence is
    # verified if does not contain all the bases
    if seq.issubset({"A", "T", "C", "G"}):
        return "DNA"
    elif seq.issubset({"A", "U", "C", "G"}):
        return "RNA"
    else:
        return "Invalid sequence"

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

def comp_st(seq):
    '''This function returns a reverse complement
    of a DNA or RNA strand'''
    seq = seq.upper()
    verified = verify(seq)
    if verified == "DNA":

        # complement strand
        seq = seq.replace("A", "t").replace(
            "C", "g").replace("T", "a").replace("G", "c")
        seq = seq.upper()
        return seq

    elif verified == "RNA":

        # complement strand
        seq = seq.replace("A", "u").replace(
            "C", "g").replace("U", "a").replace("G", "c")
        seq = seq.upper()

        # reverse strand
        return seq
    else:
        return "Invalid sequence"


def rev_comp_st(seq):
    '''This function returns a reverse complement
    of a DNA or RNA strand'''
    seq = seq.upper()
    verified = verify(seq)
    if verified == "DNA":

        # complement strand
        seq = seq.replace("A", "t").replace(
            "C", "g").replace("T", "a").replace("G", "c")
        seq = seq.upper()

        # reverse strand
        seq = seq[::-1]
        return seq

    elif verified == "RNA":

        # complement strand
        seq = seq.replace("A", "u").replace(
            "C", "g").replace("U", "a").replace("G", "c")
        seq = seq.upper()

        # reverse strand
        seq = seq[::-1]
        return seq
    else:
        return "Invalid sequence"


def len_trim(data_frame, pirna_max_len=35):
    for i in range(0, data_frame.shape[0]):
        pirna = data_frame.iloc[i, 0]
        mrna = data_frame.iloc[i, 1]
        if len(pirna) > 35:
            pirna = pirna[0:35]
            mrna = mrna[0:35]
            data_frame.iloc[i, 0] = pirna
            data_frame.iloc[i, 1] = mrna
    return data_frame

    
if '__main__' == __name__:
    '''
        *** USAGE ***
        python main.py --input_file_name test_dataset.csv
    '''
    parser = argparse.ArgumentParser(description='piRNA targeting prediction from csv file')
    parser.add_argument('--input_file_name', type=str)
    parser.add_argument('--thres', type=float)
    parser.add_argument('-b', action='store_true',
                        default=False,
                        dest='bam_t',
                        help='Input is bam file')
    parser.add_argument('--fa_name', type=str,
                        help='required by -b option')
    opt = parser.parse_args()
    thres=opt.thres
    load_model = torch.load('final_model.pth')
    model = get_model(load_model=load_model)
    print(os.getcwd())
    if opt.bam_t:
        samfile = pysam.AlignmentFile(opt.input_file_name, "rb")
        if not opt.fa_name:
            parser.error("Error: When -b is provided, --fa is also required.")
        fa = pysam.FastaFile(opt.fa_name)
        allreads = samfile.fetch()
        potential_target_tuples = []
        for i in allreads:
            if i.flag == 16:
                reference_sequence = i.get_reference_sequence().upper()[::-1]
            elif i.flag == 0:
                reference_sequence = comp_st(i.get_reference_sequence())
            else:
                continue
            piRNAname = re.sub("\\..*", "", i.qname)
            targetName = re.sub("\\..*", "", i.reference_name)
            piRNASequence = fa.fetch(piRNAname)
            strand = i.flag
            potential_target_tuples.append((piRNASequence, reference_sequence, piRNAname, targetName, strand))
        Potential_PredictTargets = pd.DataFrame(potential_target_tuples,
                                                columns=["piRNASequence", "mRNATargetSequence", "piRNAname",
                                                         "targetName", "strand"])
        Potential_PredictTargets = len_trim(Potential_PredictTargets, pirna_max_len=35)
        seq = data_preprocessing(Potential_PredictTargets, have_ans=False, pirna_max_len=35)
        prob_all = []
        model.eval()
        with torch.no_grad():
            for data in seq:
                data = torch.unsqueeze(data, 0)
                y = model(data)[0].detach()
                output = model(data)[0]
                curr_pred = [1 if i > thres else 0 for i in F.softmax(output, dim=-1)[:, 1]]
                # print(curr_pred)
                prob_all.extend(curr_pred)
        PredictTargets = Potential_PredictTargets[[i == 1 for i in prob_all]]
        PredictTargets.to_csv("target.csv")

        PredictNonTargets = Potential_PredictTargets[[i == 0 for i in prob_all]]
        PredictNonTargets.to_csv("non_target.csv")
    else:
        input_data = read_from_csv(file=opt.input_file_name)
        piRNA_max_len = max(input_data.iloc[:, 1].apply(len, 1))
        seq = data_preprocessing(data_frame=input_data, have_ans=False, pirna_max_len=piRNA_max_len)
        prediction(model=model, seq=seq, thres=thres)
