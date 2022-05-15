from keras import layers as L
from keras import backend as K
import keras
import tensorflow as tf
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from tools.encoding import one_hot_encoder
import data_parser as dp
import sys

sys.path.insert(0,"../data/")
sys.path.insert(0,"../")
sys.path.insert(0,"../tools/")

light_len = 200
heavy_len = 250
print(len(tf.config.list_physical_devices('GPU'))>0)


#LSTM = tf.keras.layers.CuDNNLSTM(16)
LSTM = tf.compat.v1.keras.layers.CuDNNLSTM(16)
#LSTM1 = tf.keras.layers.CuDNNLSTM(16,return_sequences = True)
LSTM1 = tf.compat.v1.keras.layers.CuDNNLSTM(16,return_sequences = True)

ReductionV2AUTO = tf.keras.losses.Reduction.AUTO

scaler = MinMaxScaler()

aa_order = ['ALA',
 'ARG',
 'ASN',
 'ASP',
 'CYS',
 'GLN',
 'GLU',
 'GLY',
 'HIS',
 'ILE',
 'LEU',
 'LYS',
 'MET',
 'PHE',
 'PRO',
 'SER',
 'THR',
 'TRP',
 'TYR',
 'VAL']

aa3_aa1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
           'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
           'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
           'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

class AminoAcidEncoder:
    def __init__(self, max_length, copy=True):
        """
        3D matrix scaling for RNN preparation with mask
        """
        self.copy = copy
        self.aa_order = list(map(lambda x: aa3_aa1[x], aa_order))
        self.max_length = max_length

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = np.zeros((len(X), self.max_length, len(aa_order)+1))
        for i in range(len(X)):
            for j in range(len(X[i])):
                try:
                    result[i, j, self.aa_order.index(X[i][j])] = 1
                except:
                    result[i, j, len(aa_order)] = 1
        return result

    def inverse_transform(self, X, y=None):
        result = list()
        for i in range(X.shape[0]):
            result_i=list()
            for j in range(self.max_length):
                idx = np.where(X[i,j]==1)[0]
                if idx.size != 0:
                    idx = int(idx)
                    if idx < len(self.aa_order):
                        result_i.append(self.aa_order[idx])
                    else:
                        result_i.append('')
            print(i, ''.join(result_i))
            result.append(''.join(result_i))
        return result



def get_loss(mask_value):

    """
    :param mask_value:
    :return:
    """

    mask_value = K.variable(mask_value, dtype=K.floatx())

    def masked_mse(y_true, y_pred):
        # find out which timesteps in `y_true` are not the padding character
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = K.expand_dims(1 - K.cast(mask, K.floatx()))

        loss = (y_true - y_pred) ** 2 * mask

        # take average w.r.t. the number of unmasked entries
        return K.sum(loss) / K.sum(mask)

    return masked_mse


def autoencoder_Titan(input_shape, compile = True):

    """
    General info

    Input   :
    Output  :
    """


    #setup the inputs and embed them.

    light_input = L.Input((light_len,input_shape), dtype='float', name='Light_Input')
    heavy_input = L.Input((heavy_len,input_shape), dtype='float', name='Heavy_Input')

    #light_embed = L.Embedding(input_dim=input_dims, output_dim=10, name='Light_Embed')(light_input)
    #heavy_embed = L.Embedding(input_dim=input_dims, output_dim=10, name='Heavy_Embed')(heavy_input)


    def encoder (inputs):

        #Recurrent layers

        light_RNN_layer = L.Bidirectional(LSTM, name='Light_bidirectional_RNN', merge_mode='sum')(inputs[0])
        heavy_RNN_layer = L.Bidirectional(LSTM, name='Heavy_bidirectional_RNN', merge_mode='sum')(inputs[1])


        #Dense layers

        light_dense_layer_1 = L.Dense(32, activation='relu', name='Light_encoder_dense_1')(light_RNN_layer)
        heavy_dense_layer_1 = L.Dense(32, activation='relu', name='Heavy_encoder_dense_1')(heavy_RNN_layer)

        #Merge layers

        merge_layer = L.merge.concatenate([light_dense_layer_1, heavy_dense_layer_1], name='Merged_layers')

        #Dense layer on merged

        merged_dense_layer_1 = L.Dense(32, activation='relu', name='Merged_encoder_dense_1')(merge_layer)

        bottleneck = L.Dense(50, name='bottleneck')(merged_dense_layer_1)

        return bottleneck

    def decoder(encoder_layer):
        merged_dense_decode_1 = L.Dense(16, activation='relu', name='Merged_decoder_dense_1')(encoder_layer)

        merged_dense_decode_2 = L.Dense(16, activation='relu', name='merged_decoder_dense2')(merged_dense_decode_1)

        outputs = []

        for name, length in zip(['Light', 'Heavy'], [light_len, heavy_len]):
            dense_decode_3 = L.Dense(32, activation='relu', name='{}_decoder_dense1'.format(name))(merged_dense_decode_2)

            repeat_vector_1 = L.RepeatVector(length, name='{}_decoder_repeat_vector1'.format(name))(dense_decode_3)

            rnn_decode = L.Bidirectional(LSTM1, merge_mode='sum', name='{}_decoder_bidirectional_rnn1'.format(name))(repeat_vector_1)

            decoded = L.Dense(input_shape, name='{}_output'.format(name))(rnn_decode)

            outputs.append(decoded)

        return outputs

    code = encoder([light_input, heavy_input])
    reconstruction = decoder(code)

    autoencoder = keras.models.Model(inputs=[light_input, heavy_input], outputs=reconstruction)
    encoder_model = keras.models.Model(inputs=[light_input, heavy_input], outputs=code)

    if compile:
        #mse_loss = tf.keras.losses.MeanSquaredError(reduction=ReductionV2AUTO, name='mean_squared_error')
        mask_mse_loss = get_loss(0)
        #masked_mse = get_loss(0)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adamax(), loss=mask_mse_loss)

    return encoder_model, autoencoder



encoder, autoencoder = autoencoder_Titan(21)

#SVG(model_to_dot(autoencoder, show_shapes=True).create(prog='dot', format='svg'))


print(autoencoder.count_params())

if __name__ == '__main__':

    light, heavy, source, name = dp.data_extract('../data/AbFv_animal_source.csv')

    #light_encoded = one_hot_encoder(light, 120)
    #light_encoded = scaler.fit_transform(light_encoded.reshape(-1, light_encoded.shape[-1])).reshape(light_encoded.shape)
    light_encoded = AminoAcidEncoder(max_length=light_len).transform(light)

    #heavy_encoded = one_hot_encoder(heavy, 140)
    #heavy_encoded = scaler.fit_transform(heavy_encoded.reshape(-1, heavy_encoded.shape[-1])).reshape(heavy_encoded.shape)

    heavy_encoded = AminoAcidEncoder(max_length=heavy_len).transform(heavy)

    autoencoder.fit([light_encoded, heavy_encoded], [light_encoded, heavy_encoded],
                         epochs=500, batch_size=32, validation_split=0.2, shuffle = True, steps_per_epoch=1700)