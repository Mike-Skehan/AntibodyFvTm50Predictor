import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from scipy.stats import uniform

def rf_class(X, y, iters, cv_num):
    """
    Random Forest Classifier
    :param X: features
    :param y: labels
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
    :param X: features
    :param y: labels
    :param iters: number of iterations
    :param cv_num: number of cross validations
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
    Gradient Boosted Classifier
    :param X: features
    :param y: labels
    :param params: hyperparameters
    :param iters: number of iterations
    :param cv_num: number of cross validations
    :return: best model
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


def rnn_classifier(X, y, num_epochs=100, batch_size=10):
    """
    Recurrent Neural Network Classifier
    :param X: features
    :param y: labels
    :param num_epochs: epochs to train the model
    :param batch_size: number of samples per gradient update
    :return: best model
    """
    rand = 28

    y = pd.DataFrame(y)
    y = y.values.ravel()

    X.columns = ['{}'.format(i) for i in range(len(X.columns))]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rand)

    X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Convert labels to one-hot encoded format
    y_train_encoded = to_categorical(y_train, num_classes=3)
    y_test_encoded = to_categorical(y_test, num_classes=3)

    model = Sequential()
    model.add(LSTM(100, kernel_initializer=glorot_uniform(seed=rand)))
    model.add(Dense(3, activation='softmax', kernel_initializer=glorot_uniform(seed=rand)))  # Softmax for multi-class
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    history = model.fit(X_train, y_train_encoded, epochs=num_epochs, batch_size=batch_size)

    scores = model.evaluate(X_test, y_test_encoded, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(y_pred_classes)
    print(y_test)

    #MCC evaluation
    MCC = matthews_corrcoef(y_test, y_pred_classes)
    print (MCC)

    target_names = ['<70', '70 - 75', '>75']
    print(classification_report(y_test, y_pred_classes, target_names=target_names))


    return model

