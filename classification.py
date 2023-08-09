import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV


def rf_class(X, y, params, iters, cv_num):
    """
    Random Forest Classifier
    :param X: features
    :param y: labels
    :return: model
    """

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # random forest classifier
    model = RandomForestClassifier()


    # randomized search
    rf_random = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=iters, cv=cv_num, verbose=2, n_jobs=-1)
    rf_random.fit(X_train, y_train)

    # return best estimator
    return rf_random.best_estimator_


    return model

def svm_classifier():
    return x

def gbt_classifier()
    return x

def rnn_classifier():
    return x


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