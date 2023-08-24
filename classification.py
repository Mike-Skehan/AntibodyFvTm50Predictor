import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def rf_class(X, y, iters, cv_num):
    """
    Random Forest Classifier
    :param X: features
    :param y: labels
    :param params: hyperparameters
    :param iters: number of iterations
    :param cv_num: number of cross validations
    :return: model
    """

    n_estimators = [int(x) for x in np.linspace(start=20, stop=200, num=20)]
    max_features = [1.0, 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    params = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=28)

    # random forest classifier
    model = RandomForestClassifier()

    # randomized search
    rf_random = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=iters, cv=cv_num, verbose=2,
                                   n_jobs=-1)
    rf_random.fit(X_train, y_train)

    best = rf_random.best_estimator_

    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)

    target_names = ['<70', '70 - 75', '>75']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # print MCC
    print("MCC: ", (matthews_corrcoef(y_test, y_pred)))

    # return best estimator
    return rf_random.best_estimator_


# svm classifier

def svm_classifier(X, y, iters, cv_num):
    """
    SVM Classifier
    :return: model
    """
    svm_params = {'C': uniform(loc=0, scale=1000), 'gamma': ['scale', 'auto'] + list(np.logspace(-5, 2, 10))}

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=28)

    # svm classifier
    model = SVC()

    # randomized search
    svm_random = RandomizedSearchCV(estimator=model, param_distributions=svm_params, n_iter=iters, cv=cv_num, verbose=2,
                                    n_jobs=-1)
    svm_random.fit(X_train, y_train)

    best = svm_random.best_estimator_

    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)

    target_names = ['<70', '70 - 75', '>75']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # print MCC
    print("MCC: ", (matthews_corrcoef(y_test, y_pred)))

    # return best estimator
    return svm_random.best_estimator_


def gbt_class(X, y, params, iters, cv_num):
    """
    Random Forest Classifier
    :param X: features
    :param y: labels
    :param params: hyperparameters
    :param iters: number of iterations
    :param cv_num: number of cross validations
    :return: model
    """

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=28)

    # gradient boosted classifier
    model = GradientBoostingClassifier()

    # randomized search
    gbt_random = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=iters, cv=cv_num, verbose=2,
                                    n_jobs=-1)
    gbt_random.fit(X_train, y_train)

    best = gbt_random.best_estimator_

    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)

    target_names = ['<70', '70 - 75', '>75']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # print MCC
    print("MCC: ", (matthews_corrcoef(y_test, y_pred)))

    # return best estimator
    return gbt_random.best_estimator_


def rnn_classifier(X,y, num_epochs=100, batch_size=10):

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

def eval_model(model, x_test, y_test):
    """

    :param model     : random forest regression model
    :param x_test    : test features
    :param y_test    : test labels
    :return          : model MAE, R-squared and pearson coefficient
    """

    predictions = model.predict(x_test)
    errors = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    pearsonr = stats.pearsonr(y_test, predictions)

    return np.mean(errors), r2, pearsonr
