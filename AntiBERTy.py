from igfold import IgFoldRunner
import torch
from tools import data_parser as dp
import pandas as pd

igfold = IgFoldRunner()


def seq2BERTy(heavy, light):

    """

    :param heavy:   AntibodyFv heavy chain sequence. string
    :param light:   AntibodyFv light chain sequence. string
    :return:        512 feature dataset encoded using antiBERTy on concatenated sequences.
    """

    embed_list = []
    length = len(light)

    for i in range(0, length):
        sequences = {
            "H": heavy[i],
            "L": light[i]
        }

        emb = igfold.embed(
            sequences=sequences,
        )
        berty = emb.bert_embs
        encoded = torch.sum(berty, dim=1)

        embed_list.append(encoded)
    final = torch.cat(embed_list, dim=0)

    return final


def bert_csv(data_file):
    """

    :param data_file: csv file containing Fv antibody sequences and Tm50 data
    :return: csv file containing antiBERTy encoded concatenated sequences.
    """
    light, heavy, temp = dp.data_extract(data_file)
    tensor = seq2BERTy(heavy, light)
    encoded_seq = pd.DataFrame(tensor.detach().numpy())

    return encoded_seq.to_csv('combined_bert_df.csv', index=False)