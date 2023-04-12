import numpy as np
from scipy.stats import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.legacy_tf_layers.core import Dense
from tensorflow.python.keras.models import Sequential


def rnn_regression(seq, temp):
    """
    :param seq: sequence of amino acids
    :param temp: temperature
    :return: model MAE, R-squared and pearson coefficient
    """
    # split the data into a training and testing set
    x_train, x_test, y_train, y_test = train_test_split(seq, temp, test_size=0.2, random_state=7)

    # create the model
    model = Sequential()
    model.add(LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam')

    # fit the model
    model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=2)

    # evaluate the model
    predictions = model.predict(x_test)
    errors = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    pearsonr = stats.pearsonr(y_test, predictions)

    return np.mean(errors), r2, pearsonr
