import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from tools import data_parser as dp
import sys
from tools.encoding import seq2vec
from tensorflow.keras import layers
from IPython.display import SVG
from tensorflow.keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

sys.path.insert(0,"../data/")
sys.path.insert(0,"../")
sys.path.insert(0,"../tools/")

dims = 100
latent_dim = 32


ReductionV2AUTO = tf.keras.losses.Reduction.AUTO

scaler = MinMaxScaler()

mse = tf.keras.losses.MeanSquaredError()
acc = tf.keras.metrics.Accuracy()

def autoencoderV2(input_shape):

    light_input = layers.Input((dims, input_shape), dtype='float', name='Light_Input')
    heavy_input = layers.Input((dims, input_shape), dtype='float', name='Heavy_Input')

    def encoder(inputs):

        light_lstm_layer = layers.LSTM(32,input_shape=(100,1))(inputs[0])
        heavy_lstm_layer = layers.LSTM(32,input_shape=(100,1))(inputs[1])

        light_dense_layer = layers.Dense(32, activation='relu', name='Light_encoder_dense_1')(light_lstm_layer)
        heavy_dense_layer = layers.Dense(32, activation='relu', name='Heavy_encoder_dense_1')(heavy_lstm_layer)

        merge_layer = layers.concatenate([light_dense_layer, heavy_dense_layer], name='Merged_layers')

        merged_dense_layer = layers.Dense(32, activation='relu', name='Merged_encoder_dense_1')(merge_layer)

        bottleneck = layers.Dense(50, name='bottleneck')(merged_dense_layer)

        return bottleneck

    def decoder(encoded_layer):
        merged_dense_decode_1 = layers.Dense(16, activation='relu', name='Merged_decoder_dense_1')(encoded_layer)
        merged_dense_decode_2 = layers.Dense(16, activation='relu', name='Merged_decoder_dense_2')(merged_dense_decode_1)

        outputs = []

        for name, length in zip(['Light', 'Heavy'], [dims, dims]):
            dense_decode_3 = layers.Dense(32, activation='relu', name='{}_decoder_dense1'.format(name))(merged_dense_decode_2)

            repeat_vector_1 = layers.RepeatVector(length, name='{}_decoder_repeat_vector1'.format(name))(dense_decode_3)

            lstm_decode = layers.LSTM(32,input_shape=(100,1), name='{}_decoder_bidirectional_rnn1'.format(name))(
                repeat_vector_1)

            decoded = layers.Dense(input_shape, name='{}_output'.format(name))(lstm_decode)

            outputs.append(decoded)

        return outputs

    code = encoder([light_input, heavy_input])
    reconstruction = decoder(code)

    autoencoder = tf.keras.Model(inputs=[light_input, heavy_input], outputs=reconstruction)
    encoder_model = tf.keras.Model(inputs=[light_input, heavy_input], outputs=code)

    if compile:

        autoencoder.compile(optimizer=tf.keras.optimizers.Adamax(), loss='mae', metrics=['acc'])

    return encoder_model, autoencoder


encoder, autoencoder = autoencoderV2(1)

if __name__ == '__main__':
    scaler = MinMaxScaler()

    plot_model(autoencoder,show_shapes = True, to_file='vec_model.png')
    SVG(model_to_dot(autoencoder, show_shapes=True).create(prog='dot', format='svg'))

    light, heavy, source, name = dp.data_extract_abY('../data/abYsis_data.csv')

    test_light, test_heavy, tm = dp.data_extract_Jain('../data/Jain_Ab_dataset.csv')

    light_vec = seq2vec(light)
    scaler.fit(light_vec)
    v_light_input = scaler.transform(light_vec)
    v_light_input = np.reshape(v_light_input, (v_light_input.shape[0], v_light_input.shape[1], 1))

    heavy_vec = seq2vec(heavy)
    h_scaler = scaler.fit(heavy_vec)
    v_heavy_input = scaler.transform(heavy_vec)
    v_heavy_input = np.reshape(v_heavy_input, (v_heavy_input.shape[0], v_heavy_input.shape[1], 1))

    history = autoencoder.fit([v_light_input, v_heavy_input],[v_light_input, v_heavy_input], epochs=10, batch_size=32, validation_split=0.2, shuffle=True)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

