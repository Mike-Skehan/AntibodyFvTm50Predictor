from igfold import IgFoldRunner
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from tools import data_parser as dp
import pandas as pd

igfold = IgFoldRunner()


def sequence_generator(heavy, light):
    """

    :param heavy: Heavy chain sequence. string
    :param light: Light chain sequence. string
    :return: dictionary of heavy and light chain sequences.
    """
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


class AntiBERTyEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer class for antiBERTy encoding.
    """
    def __init__(self):
        self.igfold = IgFoldRunner()

    @classmethod
    def sequence_generator(cls, X):
        """

        :param X: heavy and light chain sequences. tuple
        :return: dictionary of heavy and light chain sequences.
        """
        heavy, light = X
        sequences = {"H": heavy, "L": light}
        yield sequences

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """

        :param X: heavy and light chain sequences. tuple
        :return: antiBERTy encoded concatenated sequences. pandas dataframe
        """
        embed_list = []
        sequence_gen = self.sequence_generator(X)

        for sequences in sequence_gen:
            emb = self.igfold.embed(
                sequences=sequences,
            )
            berty = emb.bert_embs
            encoded = torch.sum(berty, dim=1)

            embed_list.append(encoded)
        final = torch.cat(embed_list, dim=0)
        encoded_seq = pd.DataFrame(final.detach().numpy())

        return encoded_seq


if __name__ == '__main__':

    seq = ('ELQMTQSPASLAVSLGQRATISCKASQSVDYDGDSYMNWYQQKPGQPPKLLIYAASNLESGIPARFSGSGSRTDFTLTINPVETDDVATYYCQQSHEDPYTFGGGTKLEIK','LESGAELVKPGASVKLSCKASGYIFTTYWMQWVKQRPGQGLEWIGEIHPSNGLTNYNEKFKSKATLTVDKSSTTAYMQLSSLTSEDSAVYYCSKGRELGRFAYWGQGTLVTVSA')

    data72 = pd.read_csv('./data/combined_datasets_72.csv')

    selected_features = data72.columns

    tensor = AntiBERTyEncoder().transform(seq)

    # encoded_seq.to_csv('./data/abYsis_bert_df.csv', index=False)
