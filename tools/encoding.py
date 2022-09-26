import numpy as np
import data_parser as dp
import sys

sys.path.insert(0,"../data/")
sys.path.insert(0,"../")

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

def one_hot_encoder(sequences, max_length):

    """
    General info

    :param sequences    : list of Fv amino acid sequences.
    :param max_length   : maximum sequence length of Fv amino acid sequence.
    :return             : amino acid sequences encoded in the one hot form.
    """
    one_hot_seq = np.zeros((len(sequences), max_length, len(amino_acids)), dtype='int')


    for x, seq in enumerate(sequences):
        for y, aa in enumerate(seq):
            loc = amino_acids.find(aa)
            if loc > 0:
                    one_hot_seq[x, y, loc] = 1
    return one_hot_seq

if __name__ == '__main__':
    #light, heavy, source, name = dp.data_extract('../data/AbFv_animal_source.csv')
    #print(type(heavy[1]))
    #VH_encoded = one_hot_encoder(heavy,140)
    #print (VH_encoded[700])


    light, heavy, source, name = dp.data_extract_abY("../data/abYsis_data.csv")

    g = np.zeros((len(heavy), 150, 20), dtype='int')

    for x, seq in enumerate(heavy):
        print(x)
        for y, aa in enumerate(seq):
            loc = amino_acids.find(aa)
            if loc > 0:
                    g[x, y, loc] = 1
    print(g)
