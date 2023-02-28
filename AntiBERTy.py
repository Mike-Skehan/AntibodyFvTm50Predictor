from igfold import IgFoldRunner
import torch
from tools import data_parser as dp
import pandas as pd

igfold = IgFoldRunner()


def sequence_generator(heavy, light):
    length = len(light)

    for i in range(0, length):
        sequences = {
            "H": heavy[i],
            "L": light[i]
        }
        yield sequences


def seq2BERTy(heavy, light):

    """

    :param heavy:   AntibodyFv heavy chain sequence. string
    :param light:   AntibodyFv light chain sequence. string
    :return:        512 feature dataset encoded using antiBERTy on concatenated sequences.
    """

    embed_list = []
    sequence_gen = sequence_generator(heavy, light)

    for sequences in sequence_gen:
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


if __name__ == '__main__':

    light, heavy, name, species = dp.data_extract_abY('./data/abYsis_data.csv')

    sequence_generator(heavy, light)
    seq2BERTy(heavy,light)
