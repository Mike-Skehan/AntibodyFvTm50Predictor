import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt


def rnn_regression(seq, temp, num_epochs=100, batch_size=10, test_size=0.1):
    """
    :param seq: sequence of amino acids
    :param temp: temperature
    :param num_epochs: number of epochs to train the model for
    :param batch_size: batch size for training the model
    :param test_size: the proportion of the dataset to include in the test split
    :return: model MAE, R-squared and pearson coefficient
    """

    # convert seq and temp to numpy arrays
    seq = seq.values.reshape((seq.shape[0], seq.shape[1], 1))
    temp = np.array(temp)

    # split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(seq, temp, test_size=test_size, random_state=42)

    # create the model
    model = Sequential()
    model.add(LSTM(100, input_shape=(seq.shape[1], seq.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam')

    # fit the model
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2, validation_split=0.2)

    # evaluate the model on the test set
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    pearsonr = np.corrcoef(y_test, predictions.T)[0, 1]

    # plot training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    return mae, r2, pearsonr
