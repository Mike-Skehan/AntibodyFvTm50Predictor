import numpy as np
import data_parser as dp
import sys
import biovec
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

    list_vec = seq2vec(light)
    array_vec = np.vstack(list_vec)

    X = array_vec
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                           init = 'random', perplexity = 100).fit_transform(X)


    dataset = pd.DataFrame({'Column1': X_embedded[:, 0], 'Column2': X_embedded[:, 1]})


    dataset['source']= name

    #print (dataset)

    tsne_plot = sns.lmplot(x="Column1", y="Column2", data=dataset, fit_reg=False, hue='source', legend=False,scatter_kws={"s": 10,'alpha': 0.5})
    plt.legend(loc='lower right')
    plt.show()

    tsne_plot.figure.savefig("TSNE_plot_per50.png", bbox_inches='tight')
