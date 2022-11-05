from keras import layers as L, Sequential
from keras import backend as K
import keras
import tensorflow as tf
from IPython.display import SVG
from keras.layers import Dense, Dropout, RepeatVector
from tensorflow.keras.utils import plot_model
import pydot
import graphviz
from keras.utils.vis_utils import model_to_dot
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM


import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
import data_parser as dp
import sys
from tools.encoding import seq2vec

sys.path.insert(0,"../data/")
sys.path.insert(0,"../")
sys.path.insert(0,"../tools/")

light_len = 100
heavy_len = 100


#LSTM = tf.compat.v1.keras.layers.CuDNNLSTM(16)

ReductionV2AUTO = tf.keras.losses.Reduction.AUTO

scaler = MinMaxScaler()

mse = tf.keras.losses.MeanSquaredError()
acc = tf.keras.metrics.Accuracy()



if __name__ == '__main__':
    scaler = MinMaxScaler()


    #plot_model(autoencoder,show_shapes = True, to_file='model.png')
    #SVG(model_to_dot(autoencoder, show_shapes=True).create(prog='dot', format='svg'))

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

    print(type(v_heavy_input))

    model = Sequential()
    model.add(LSTM(32,input_shape=(100,1)))
    model.add(Dense(1))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(100))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    history = model.fit(v_heavy_input, v_heavy_input, epochs=100, batch_size=32, validation_split=0.1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()


    #history = autoencoder.fit([v_light_input, v_heavy_input], [v_light_input, v_heavy_input],
    #                     epochs=1000, batch_size=32, validation_split=0.2, shuffle = True)



    #plt.figure(figsize=(10, 4))
    #plt.plot(autoencoder.history.history['val_loss'], label='Validation loss')
    #plt.plot(autoencoder.history.history['loss'], label='Training loss')
    #plt.legend()
    #plt.grid()
    #plt.tight_layout()
    #plt.savefig('vec_loss_output_testing.png')

    #print("Evaluate on test data")
    #results = autoencoder.evaluate([test_light_encoded, test_heavy_encoded], [test_light_encoded, test_heavy_encoded], batch_size=32)

    #print (results)
    #print(dict(zip(autoencoder.metrics_names, results)))