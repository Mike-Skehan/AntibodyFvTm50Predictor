import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
import pandas as pd


def rnn_cv(X,y, num_epochs=100, batch_size=10):

    """

    :param X: encoded sequences
    :param y: labels
    :param num_epochs: epochs to train the model
    :param batch_size: number of samples per gradient update
    :return: best model
    """

    sns.set_style(style='white')

    rand = 28

    y = pd.DataFrame(y)
    y = y.values.ravel()

    X.columns = ['{}'.format(i) for i in range(len(X.columns))]

    # set up train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rand)

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    axs = axs.flatten()

    mae_scores = []
    r2_scores = []
    pearsonr_scores = []

    # set up the k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_index, val_index) in enumerate(kf.split(X_train)):

        k_X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        k_y_train, y_val = y_train[train_index], y_train[val_index]

        # create the RNN model
        k_X_train = k_X_train.values.reshape((k_X_train.shape[0], k_X_train.shape[1], 1))
        X_val = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))
        k_y_train = np.array(k_y_train)
        y_val = np.array(y_val)
        model = Sequential()
        model.add(LSTM(100, input_shape=(k_X_train.shape[1], k_X_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_absolute_error', optimizer='adam')

        # fit the model
        history = model.fit(k_X_train, k_y_train, epochs=num_epochs, batch_size=batch_size, verbose=2, validation_data=
        (X_val, y_val))

        # evaluate the model on the test set
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # y_test and predictions are arrays with shape (36, 1)
        y_test_reshaped = np.squeeze(y_test)
        predictions_reshaped = np.squeeze(predictions)

        # Calculate Pearson correlation coefficient
        pearson, _ = pearsonr(y_test_reshaped, predictions_reshaped)

        mae_scores.append(mae)
        r2_scores.append(r2)
        pearsonr_scores.append(pearson)

        # plot training and validation loss on the appropriate subplot
        axs[i].plot(history.history['loss'])
        axs[i].plot(history.history['val_loss'])
        axs[i].set_title(f'Fold {i+1} Loss')
        axs[i].set_ylabel('Loss')
        axs[i].set_xlabel('Epoch')
        axs[i].legend(['Train', 'Validation'], loc='upper right')

    # adjust the layout and display the plots
    plt.tight_layout()
    plt.show()

    print('MAE: %.3f (%.3f)' % (np.mean(mae_scores), np.std(mae_scores)))
    print('R2: %.3f (%.3f)' % (np.mean(r2_scores), np.std(r2_scores)))
    print('Pearsonr: %.3f (%.3f)' % (np.mean(pearsonr_scores), np.std(pearsonr_scores)))

    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, predictions)
    plt.title(f'Predictions vs. Actual\nMAE={(np.mean(mae_scores)):.2f}, R2={(np.mean(r2_scores)):.2f}, Pearsonr={(np.mean(pearsonr_scores)):.2f}')
    plt.ylabel('Actual')
    plt.xlabel('Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.show()

    return model

