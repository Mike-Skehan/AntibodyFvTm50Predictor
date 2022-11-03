import numpy as np
import data_parser as dp
import sys
import biovec


sys.path.insert(0,"../data/")
sys.path.insert(0,"../")

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

def one_hot_encoder(sequences, max_length):

    """
    General info

    :param sequences    : list of Fv amino acid sequences.
    :param max_length   : maximum sequence length of Fv amino acid sequence.
    :return             : amino acid sequences encoded in the one hot form in a 3D array.
    """
    one_hot_seq = np.zeros((len(sequences), max_length, len(amino_acids)), dtype='int')


    for x, seq in enumerate(sequences):
        for y, aa in enumerate(seq):
            loc = amino_acids.find(aa)
            if loc > 0:
                    one_hot_seq[x, y, loc] = 1
    return one_hot_seq

def seq2vec(seq_list):

    """
    General info

    :param seq_list     : list of Fv amino acid sequences.
    :return             : amino acid sequences encoded into a 100D array using .
    """

    pv = biovec.models.load_protvec('./swissprot-reviewed-protvec.model')
    light_vec = []
    for seq in seq_list:
        vec = sum(pv.to_vecs(seq))
        light_vec.append(vec)
    return light_vec



if __name__ == '__main__':

    light, heavy, name, source = dp.data_extract_abY('../data/abYsis_data.csv')

    print (seq2vec(light)[5])

