import numpy as np
import data_parser as dp
import sys
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.sequence import skipgrams, pad_sequences, make_sampling_table
from keras.preprocessing.text import hashing_trick
from keras.layers import Embedding, Input, Reshape, Dense, merge
from keras.models import Sequential, Model
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import csv

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



if __name__ == '__main__':

    light, heavy, name, source = dp.data_extract_abY('../data/abYsis_data.csv')

    #Load Ehsan Asgari's embeddings
    ehsanEmbed = []
    with open("../data/protVec_100d_3grams.csv") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            ehsanEmbed.append(line[0].split('\t'))
    threemers = [vec[0] for vec in ehsanEmbed]
    embeddingMat = [[float(n) for n in vec[1:]] for vec in ehsanEmbed]
    threemersidx = {}  # generate word to index translation dictionary. Use for kmersdict function arguments.

    for i, kmer in enumerate(threemers):
        threemersidx[kmer] = i

    # Set parameters
    vocabsize = len(threemersidx)
    window_size = 25
    num_cores = multiprocessing.cpu_count()  # For parallel computing


    def kmerlists(seq):
        kmer0 = []
        kmer1 = []
        kmer2 = []
        for i in range(0, len(seq) - 2, 3):
            if len(seq[i:i + 3]) == 3:
                kmer0.append(seq[i:i + 3])
            i += 1
            if len(seq[i:i + 3]) == 3:
                kmer1.append(seq[i:i + 3])
            i += 1
            if len(seq[i:i + 3]) == 3:
                kmer2.append(seq[i:i + 3])
        return [kmer0, kmer1, kmer2]


    # Same as kmerlists function but outputs an index number assigned to each kmer. Index number is from Asgari's embedding
    def kmersindex1(seqs, kmersdict=threemersidx):
        kmers = []
        for i in range(len(seqs)):
            kmers.append(kmerlists(seqs[i]))
        kmers = np.array(kmers).flatten().flatten(order='F')
        kmersindex = []
        for seq in kmers:
            temp = []
            for kmer in seq:
                try:
                    temp.append(kmersdict[kmer])
                except:
                    temp.append(kmersdict[''])
            kmersindex.append(temp)
        return kmersindex

    def kmersindex(seqs, kmersdict=threemersidx):
        kmersindex = []
        x = kmerlists(seqs)
        for seq in x:
            temp = []
            for kmer in seq:
                try:
                    temp.append(kmersdict[kmer])
                except:
                    temp.append(kmersdict[''])
            kmersindex.append(temp)
        return(kmersindex)

    x = kmerlists(light[0])


    sampling_table = make_sampling_table(vocabsize)


    def generateskipgramshelper(kmersindicies):
        couples, labels = skipgrams(kmersindicies, vocabsize, window_size=window_size, sampling_table=sampling_table)
        if len(couples) == 0:
            couples, labels = skipgrams(kmersindicies, vocabsize, window_size=window_size,
                                        sampling_table=sampling_table)
        if len(couples) == 0:
            couples, labels = skipgrams(kmersindicies, vocabsize, window_size=window_size,
                                        sampling_table=sampling_table)
        else:
            word_target, word_context = zip(*couples)
            return word_target, word_context, labels


    def generateskipgrams(seqs, kmersdict=threemersidx):
        kmersidx = kmersindex(seqs, kmersdict)
        return Parallel(n_jobs=num_cores)(delayed(generateskipgramshelper)(kmers) for kmers in kmersidx)


    print("Sample sequence")
    print(light[0])
    print("")
    print("Convert sequence to list of kmers")
    print(kmerlists(light[0]))
    print("")
    print("Convert kmers to their index on the embedding")
    print(kmersindex(light[0]))
    #print("")



    #testskipgrams = generateskipgrams(light[0])
    #print("Sample skipgram input:")
    #print("Word Target:", testskipgrams[0][0][0])
    #print("Word Context:", testskipgrams[0][1][0])
    #print("Label:", testskipgrams[0][2][0])

    #print(kmersindex(light[0], threemersidx))

    # create some input variables
    input_target = Input((1,))
    input_context = Input((1,))
    vector_dim = 100

    embedding = Embedding(vocabsize, vector_dim, input_length=1, name='embedding')
    embedding.build((None,))
    embedding.set_weights(np.array([embeddingMat]))  # Load Asgari's embedding as initial weights

    target = embedding(input_target)
    target = Reshape((vector_dim, 1))(target)
    context = embedding(input_context)
    context = Reshape((vector_dim, 1))(context)


    def protvec(kmersdict, seq, embeddingweights=embeddingMat):
        # Convert seq to three lists of kmers
        kmerlist = kmerlists(seq)
        kmerlist = [j for i in kmerlist for j in i]
        # Convert center kmers to their vector representations
        kmersvec = [0] * 100
        for kmer in kmerlist:
            try:
                kmersvec = np.add(kmersvec, embeddingweights[kmersdict[kmer]])
            except:
                kmersvec = np.add(kmersvec, embeddingweights[kmersdict['']])
        return kmersvec


    def formatprotvecs(protvecs):
        # Format protvecs for classifier inputs by transposing the matrix
        protfeatures = []
        for i in range(100):
            protfeatures.append([vec[i] for vec in protvecs])
        protfeatures = np.array(protfeatures).reshape(len(protvecs), len(protfeatures))
        return protfeatures
ls

    def formatprotvecsnormalized(protvecs):
        # Formatted protvecs with feature normalization
        protfeatures = []
        for i in range(100):
            tempvec = [vec[i] for vec in protvecs]
            mean = np.mean(tempvec)
            var = np.var(tempvec)
            protfeatures.append([(vec[i] - mean) / var for vec in protvecs])
        protfeatures = np.array(protfeatures).reshape(len(protvecs), len(protfeatures))
        return protfeatures


    def sequences2protvecsCSV(filename, seqs, kmersdict=threemersidx, embeddingweights=embeddingMat):
        # Convert a list of sequences to protvecs and save protvecs to a csv file
        # ARGUMENTS;
        # filename: string, name of csv file to save to, i.e. "sampleprotvecs.csv"
        # seqs: list, list of amino acid sequences
        # kmersdict: dict to look up index of kmer on embedding, default: Asgari's embedding index
        # embeddingweights: 2D list or np.array, embedding vectors, default: Asgari's embedding vectors

        swissprotvecs = Parallel(n_jobs=num_cores)(delayed(protvec)(kmersdict, seq, embeddingweights) for seq in seqs)
        swissprotvecsdf = pd.DataFrame(formatprotvecs(swissprotvecs))
        swissprotvecsdf.to_csv(filename, index=False)
        return swissprotvecsdf


    sequences2protvecsCSV("testprotvecs.csv", light[:5])