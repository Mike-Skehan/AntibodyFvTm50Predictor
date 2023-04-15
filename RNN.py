import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def rnn_cv():
    # set up the subplots
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    axs = axs.flatten()

    mae_scores = []
    r2_scores = []
    pearsonr_scores = []

    # set up the k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(kf.split(X)):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # create the RNN model
        X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        model = Sequential()
        model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_absolute_error', optimizer='adam')

        # fit the model
        history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2, validation_data=(X_test, y_test))

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

    return model

