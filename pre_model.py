import torch
import pandas as pd


def read_from_csv(file = "/Users/zhangtianjiao/work/piRNA_target/predict/input/Demo.csv"):
    data_frame = pd.read_csv(file)
    return data_frame


def get_unmatched_num(data_frame):
    """
    Returns the number of unmatched complementary pairs in each piRNA-target pair

    :param data_frame: pandas dataframe: First column piRNA sequence , second column mRNA sequence
    (eg:pd.DataFrame({"piRNASequence" : ["TGAAATGTAAATGAAGAAAATACCTAATTA","TGTAGTACTTTTTGAAGTCTTT"],
    "mRNATargetSequence" : ["CATTTACTTCTTTTATTGATTATTTTTTAA","TGAAAGACTTCAGAAGTGTGTG"]}))

    :return: A list containing the number of unmatched complementary pairs of each piRNA-mRNA pair (eg:[15,4])
    """
    complementary_dict = {('A', 'T'): "Paired", ('T', 'A'): "Paired", ('G', 'C'): "Paired", ('C', 'G'): "Paired",
                          ('A', 'A'): "UnPaired", ('T', 'T'): "UnPaired", ('G', 'G'): "UnPaired",
                          ('C', 'C'): "UnPaired",
                          ('A', 'C'): "UnPaired", ('T', 'C'): "UnPaired", ('G', 'A'): "UnPaired",
                          ('C', 'T'): "UnPaired",
                          ('A', 'G'): "UnPaired", ('T', 'G'): "SemiPaired", ('G', 'T'): "SemiPaired",
                          ('C', 'A'): "UnPaired",
                          ('A', 'N'): "UnPaired", ('T', 'N'): "UnPaired", ('G', 'N'): "UnPaired",
                          ('C', 'N'): "UnPaired",
                          ('N', 'A'): "UnPaired", ('N', 'T'): "UnPaired", ('N', 'G'): "UnPaired", ('N', 'C'): "UnPaired"
                          }
    data_list = []
    for i in range(0, data_frame.shape[0]):
        pirna = data_frame.iloc[i, 0]
        mrna = data_frame.iloc[i, 1]
        tmplist = [complementary_dict[i] for i in zip(pirna, mrna)]
        data_list.append(tmplist)
    unmatched = []
    for i in data_list:
        unmatched.append(sum([j == "UnPaired" for j in i]))
    return unmatched


def get_matched_num(data_frame):
    """
    Returns the number of matched complementary pairs in each piRNA-target pair

    :param data_frame: pandas dataframe: First column piRNA sequence , second column mRNA sequence
    (eg:pd.DataFrame({"piRNASequence" : ["TGAAATGTAAATGAAGAAAATACCTAATTA","TGTAGTACTTTTTGAAGTCTTTT"],
    "mRNATargetSequence" : ["CATTTACTTCTTTTATTGATTATTTTTTAA","TGAAAGACTTCAGAAGTGTGTG"]}))

    :return: A list containing the number of matched complementary pairs of each piRNA-mRNA pair (eg:[13,2])
    """
    complementary_dict = {('A', 'T'): "Paired", ('T', 'A'): "Paired", ('G', 'C'): "Paired", ('C', 'G'): "Paired",
                          ('A', 'A'): "UnPaired", ('T', 'T'): "UnPaired", ('G', 'G'): "UnPaired",
                          ('C', 'C'): "UnPaired",
                          ('A', 'C'): "UnPaired", ('T', 'C'): "UnPaired", ('G', 'A'): "UnPaired",
                          ('C', 'T'): "UnPaired",
                          ('A', 'G'): "UnPaired", ('T', 'G'): "SemiPaired", ('G', 'T'): "SemiPaired",
                          ('C', 'A'): "UnPaired",
                          ('A', 'N'): "UnPaired", ('T', 'N'): "UnPaired", ('G', 'N'): "UnPaired",
                          ('C', 'N'): "UnPaired",
                          ('N', 'A'): "UnPaired", ('N', 'T'): "UnPaired", ('N', 'G'): "UnPaired", ('N', 'C'): "UnPaired"
                          }
    data_list = []
    for i in range(0, data_frame.shape[0]):
        pirna = data_frame.iloc[i, 0]
        mrna = data_frame.iloc[i, 1]
        tmplist = [complementary_dict[i] for i in zip(pirna, mrna)]
        data_list.append(tmplist)
    matched = []
    for i in data_list:
        matched.append(sum([j == "Paired" for j in i]))
    return matched


def data_preprocessing_stats(data_frame, gu_tolerance=True):
    """
    :param data_frame:  data_frame: pandas dataframe: First column piRNA sequence , second column mRNA sequence
    (eg:pd.DataFrame({"piRNASequence" : ["TGAAATGTAAATGAAGAAAATACCTAATTA","TGTAGTACTTTTTGAAGTCTTTT"],
    "mRNATargetSequence" : ["CATTTACTTCTTTTATTGATTATTTTTTAA","TGAAAGACTTCAGAAGTGTGTG"]}))
    :param gu_tolerance: Whether or not GU was considered a half-match
    :return: sequence list (e.g. ['UnPairedUnPairedPairedPairedPairedPairedPairedUnPairedPairedUnPairedPairedUnPairedSemiPairedPairedUnPairedSemiPairedPairedUnPairedUnPairedPairedUnPairedUnPairedUnPairedUnPairedUnPairedPairedPairedUnPairedPairedUnPaired',
    'UnPairedUnPairedPairedUnPairedUnPairedSemiPairedUnPairedUnPairedUnPairedUnPairedUnPairedPairedSemiPairedUnPairedUnPairedUnPairedSemiPairedSemiPairedUnPairedSemiPairedUnPairedSemiPaired']])
    """
    if not gu_tolerance:
        complementary_dict = {('A', 'T'): "Paired", ('T', 'A'): "Paired", ('G', 'C'): "Paired", ('C', 'G'): "Paired",
                              ('A', 'A'): "UnPaired", ('T', 'T'): "UnPaired", ('G', 'G'): "UnPaired",
                              ('C', 'C'): "UnPaired",
                              ('A', 'C'): "UnPaired", ('T', 'C'): "UnPaired", ('G', 'A'): "UnPaired",
                              ('C', 'T'): "UnPaired",
                              ('A', 'G'): "UnPaired", ('T', 'G'): "UnPaired", ('G', 'T'): "UnPaired",
                              ('C', 'A'): "UnPaired",
                              ('A', 'N'): "UnPaired", ('T', 'N'): "UnPaired", ('G', 'N'): "UnPaired",
                              ('C', 'N'): "UnPaired",
                              ('N', 'A'): "UnPaired", ('N', 'T'): "UnPaired", ('N', 'G'): "UnPaired",
                              ('N', 'C'): "UnPaired"
                              }

    else:
        complementary_dict = {('A', 'T'): "Paired", ('T', 'A'): "Paired", ('G', 'C'): "Paired", ('C', 'G'): "Paired",
                              ('A', 'A'): "UnPaired", ('T', 'T'): "UnPaired", ('G', 'G'): "UnPaired",
                              ('C', 'C'): "UnPaired",
                              ('A', 'C'): "UnPaired", ('T', 'C'): "UnPaired", ('G', 'A'): "UnPaired",
                              ('C', 'T'): "UnPaired",
                              ('A', 'G'): "UnPaired", ('T', 'G'): "SemiPaired", ('G', 'T'): "SemiPaired",
                              ('C', 'A'): "UnPaired",
                              ('A', 'N'): "UnPaired", ('T', 'N'): "UnPaired", ('G', 'N'): "UnPaired",
                              ('C', 'N'): "UnPaired",
                              ('N', 'A'): "UnPaired", ('N', 'T'): "UnPaired", ('N', 'G'): "UnPaired",
                              ('N', 'C'): "UnPaired"
                              }
    data_list = []
    for i in range(0, data_frame.shape[0]):
        pirna = data_frame.iloc[i, 0]
        mrna = data_frame.iloc[i, 1]
        tmplist = ''.join([complementary_dict[i] for i in zip(pirna, mrna)])
        data_list.append(tmplist)
    return data_list







def data_preprocessing(data_frame, have_ans=False, pirna_max_len=34, gu_tolerance=True):
    """
    :param data_frame:  data_frame: pandas dataframe: First column piRNA sequence , second column mRNA sequence
    (eg:pd.DataFrame({"piRNASequence" : ["TGAAATGTAAATGAAGAAAATACCTAATTA","TGTAGTACTTTTTGAAGTCTTT"],
    "mRNATargetSequence" : ["CATTTACTTCTTTTATTGATTATTTTTTAA","TGAAAGACTTCAGAAGTGTGTG"]}))
    :param have_ans: data_frame contains a column "Label" (e.g.[0,1,1,0];1 represents the target and 0 vice versa)
    :param pirna_max_len: maximum length of piRNA
    :param gu_tolerance: Whether or not GU was considered a half-match
    :return: torch tensor(s)
    """
    if not gu_tolerance:
        onehot = {ele: ([0]*idx + [1] + [0]*(1-idx)) for idx, ele in enumerate(['UnPaired', 'Paired'])}
        complementary_dict = {('A', 'T'): "Paired", ('T', 'A'): "Paired", ('G', 'C'): "Paired", ('C', 'G'): "Paired",
                              ('A', 'A'): "UnPaired", ('T', 'T'): "UnPaired", ('G', 'G'): "UnPaired",
                              ('C', 'C'): "UnPaired",
                              ('A', 'C'): "UnPaired", ('T', 'C'): "UnPaired", ('G', 'A'): "UnPaired",
                              ('C', 'T'): "UnPaired",
                              ('A', 'G'): "UnPaired", ('T', 'G'): "UnPaired", ('G', 'T'): "UnPaired",
                              ('C', 'A'): "UnPaired",
                              ('A', 'N'): "UnPaired", ('T', 'N'): "UnPaired", ('G', 'N'): "UnPaired",
                              ('C', 'N'): "UnPaired",
                              ('N', 'A'): "UnPaired", ('N', 'T'): "UnPaired", ('N', 'G'): "UnPaired",
                              ('N', 'C'): "UnPaired"
                              }

    else:
        onehot = {ele: ([0] * idx + [1] + [0] * (2 - idx)) for idx, ele in
                  enumerate(['UnPaired', "SemiPaired", 'Paired'])}
        complementary_dict = {('A', 'T'): "Paired", ('T', 'A'): "Paired", ('G', 'C'): "Paired", ('C', 'G'): "Paired",
                              ('A', 'A'): "UnPaired", ('T', 'T'): "UnPaired", ('G', 'G'): "UnPaired",
                              ('C', 'C'): "UnPaired",
                              ('A', 'C'): "UnPaired", ('T', 'C'): "UnPaired", ('G', 'A'): "UnPaired",
                              ('C', 'T'): "UnPaired",
                              ('A', 'G'): "UnPaired", ('T', 'G'): "SemiPaired", ('G', 'T'): "SemiPaired",
                              ('C', 'A'): "UnPaired",
                              ('A', 'N'): "UnPaired", ('T', 'N'): "UnPaired", ('G', 'N'): "UnPaired",
                              ('C', 'N'): "UnPaired",
                              ('N', 'A'): "UnPaired", ('N', 'T'): "UnPaired", ('N', 'G'): "UnPaired",
                              ('N', 'C'): "UnPaired"
                              }
    x = []
    data_list = []
    for i in range(0, data_frame.shape[0]):
        pirna = data_frame.iloc[i, 0]
        mrna = data_frame.iloc[i, 1]
        tmplist = [complementary_dict[i] for i in zip(pirna, mrna)]
        data_list.append(tmplist)
    for m in data_list:
        # padding_matrix = [[0,0] for i in range(piRNA_max_len - len(m))]
        # padding_matrix = [[0, 0,0] for i in range(piRNA_max_len - len(m))]
        padding_matrix = [[0] * len(onehot['Paired']) for i in range(pirna_max_len - len(m))]
        tmp_one_hot = [onehot[c] for c in m] + padding_matrix
        x.append(tmp_one_hot)

    tensor_x = torch.tensor(x).float().to()
    if have_ans:
        y = [i for i in data_frame['Label']]
        tensor_y = torch.tensor(y).int().to()
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
    data = inputs if batch_first is True else inputs.permute(1, 0, 2)
    for i in data:
        ind = [not x.equal(torch.Tensor([0.] * inputs.shape[-1])) for x in i]
        len_list.append(len(i[ind]))
    return len_list