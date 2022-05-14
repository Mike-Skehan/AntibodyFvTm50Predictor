from keras import layers as L
import keras
import tensorflow as tf
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from tools.encoding import one_hot_encoder
import data_parser as dp
import sys

sys.path.insert(0,"../data/")
sys.path.insert(0,"../")
sys.path.insert(0,"../tools/")

light_len = 140
heavy_len = 120

LSTM = tf.keras.layers.LSTM(16)
LSTM1 = tf.keras.layers.LSTM(16,return_sequences = True)

ReductionV2AUTO = tf.keras.losses.Reduction.AUTO

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

        bottleneck = L.Dense(2, name='bottleneck')(merged_dense_layer_1)

        return bottleneck

    def decoder(encoder_layer):
        merged_dense_decode_1 = L.Dense(32, activation='relu', name='Merged_decoder_dense_1')(encoder_layer)

        merged_dense_decode_2 = L.Dense(64, activation='relu', name='merged_decoder_dense2')(merged_dense_decode_1)

        outputs = []

        for name, length in zip(['Light', 'Heavy'], [light_len, heavy_len]):
            dense_decode_3 = L.Dense(16, activation='relu', name='{}_decoder_dense1'.format(name))(merged_dense_decode_2)

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
        mse_loss = tf.keras.losses.MeanSquaredError(reduction=ReductionV2AUTO, name='mean_squared_error')
        #masked_mse = get_loss(0)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adamax(), loss=mse_loss)

    return encoder_model, autoencoder



encoder, autoencoder = autoencoder_Titan(4)

#SVG(model_to_dot(autoencoder, show_shapes=True).create(prog='dot', format='svg'))


print(autoencoder.count_params())

if __name__ == '__main__':
    light, heavy, source, name = dp.data_extract('../data/AbFv_animal_source.csv')
    light_encoded = one_hot_encoder(light, 140)
    heavy_encoded = one_hot_encoder(heavy, 140)
    autoencoder.fit([light_encoded, heavy_encoded], [light_encoded, heavy_encoded],
                          epochs=2000, batch_size=32, validation_split=0.2)