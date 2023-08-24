import sys

from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import statistics as st
from scipy.stats import uniform
import seaborn as sns

sys.path.insert(0, "../data/")
sys.path.insert(0, "../")
sys.path.insert(0, "../tools")


def random_search(x_train, y_train, params, iters, cv_num):
    """

    :param x_train  : training features
    :param y_train  : training labels
    :param params   : random search parameters
    :param iters    : number of iteration for the search
    :param cv_num   : number of cross validations
    :return         : model with the best accuracy
    """

    rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=params, n_iter=iters,
                                   cv=cv_num, verbose=2, n_jobs=-1)

    rf_random.fit(x_train, y_train)

    return rf_random.best_estimator_


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


def eval_avg(y_test):
    """

    :param y_test  : test labels
    :return        : mean absolute error from comparing label to average of labels.
    """

    avg_temp = np.mean(y_test)
    avg_list = []
    for i in range(len(y_test)):
        avg_list.append(avg_temp)
    errors = mean_absolute_error(y_test, avg_list)
    return np.mean(errors)


def compare_to_avg(model, x_test, y_test):
    """
    :param model    : random forest regression model
    :param x_test   : test features
    :param y_test   : test labels
    :return         : Comparison between the model accuracy and accuracy from taking the label mean.
    """
    model_mae = eval_model(model, x_test, y_test)
    avg_mae = eval_avg(y_test)
    improvement = ((avg_mae - model_mae[0]) / avg_mae) * 100

    return 'Model error  : {:0.2f} C\nAverage error: {:0.2f} C\nImprovement  : {:0.2f}%'.format(float(model_mae[0]),
                                                                                                avg_mae,
                                                                                                improvement)


def save_result(model, dataset, model_name, model_loc, pearson_result):
    """

    :param model            : random forest regression model
    :param dataset          : the test dataset used
    :param model_name       : model identifier in format: ab_rf_[ddmmyyyy]_[r2 score].joblib
    :param model_loc        : model file path
    :param pearson_result   : pearson coefficient score
    :return                 : logs regression result to csv file
    """

    try:
        df = pd.read_csv('../models/results.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Dataset', 'Model', 'Model File', 'Features File', 'MAE', 'Pearson Coeff'])
    joblib.dump(model, model_loc)
    new_data = {
        'Dataset': [dataset],
        'Model': [model_name],
        'Model File': [model_loc],
        'Pearson': [pearson_result]
    }

    df = df.append(new_data, ignore_index=True)

    df.to_csv('../models/results.csv', index=False)


def rf_kfold(X, y, params, iters, cv_num, k):
    # Create a random forest regressor
    rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=params, n_iter=iters,
                                   cv=cv_num, verbose=2, n_jobs=-1)

    # Create a k-fold cross-validator with 5 folds
    kf = KFold(n_splits=k)

    # Use cross_val_score to get the scores for each fold
    scores = cross_val_score(rf_random.best_estimator_, X, y, cv=kf)

    # Print the mean and standard deviation of the scores
    print("Mean R-squared score:", scores.mean())
    print("Standard deviation of R-squared scores:", scores.std())

    return scores.mean(), scores.std()


def rf_cv(X, y):
    """

    :param X: Features
    :param y: Labels
    :return: Best model
    """
    sns.set_style(style='white')
    rand = 28

    y = pd.DataFrame(y)
    y = y.values.ravel()

    X.columns = ['{}'.format(i) for i in range(len(X.columns))]

    # set up train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rand)

    # set up the subplots
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 30))
    axs = axs.flatten()
    plt.subplots_adjust(hspace=0.5)

    mae_scores = []
    r2_scores = []
    pearsonr_scores = []

    # set up the k-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=rand)

    best_model = None
    best_score = -np.inf

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

    for i, (train_index, val_index) in enumerate(kf.split(X_train)):

        k_X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        k_y_train, y_val = y_train[train_index], y_train[val_index]

        # create the SVM model
        model = RandomForestRegressor()

        # set up the randomized search
        random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=20,
                                           cv=10, verbose=2, n_jobs=-1)

        random_search.fit(k_X_train, k_y_train)

        # evaluate the model on the validation set
        predictions = random_search.predict(X_val)
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)

        # y_val and predictions are arrays with shape (36, 1)
        y_val_reshaped = np.squeeze(y_val)
        predictions_reshaped = np.squeeze(predictions)

        # Calculate Pearson correlation coefficient
        pearsonr = stats.pearsonr(y_val_reshaped, predictions_reshaped)

        # keep track of the best model based on validation score
        if r2 > best_score:
            best_score = r2
            best_model = model

        mae_scores.append(mae)
        r2_scores.append(r2)
        pearsonr_scores.append(pearsonr[0])

        # scatter plot predictions vs. actual values on the appropriate subplot
        axs[i].scatter(y_val, predictions)
        axs[i].set_title(f'Fold {i + 1} Predictions vs. Actual\nMAE={mae:.2f}, R2={r2:.2f}, Pearsonr={pearsonr[0]:.2f}')
        axs[i].set_ylabel('Actual')
        axs[i].set_xlabel('Predictions')
        axs[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

    # print the model's mean and standard dev performance metrics
    print('Validation MAE: %.3f (%.3f)' % ((st.mean(mae_scores)), st.stdev(mae_scores)))
    print('Validation R2: %.3f (%.3f)' % ((st.mean(r2_scores)), st.stdev(r2_scores)))
    print('Validation Pearsonr: %.3f (%.3f)' % ((st.mean(pearsonr_scores)), (st.stdev(pearsonr_scores))))

    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    y_pred_reshaped = np.squeeze(y_pred)
    y_test_reshaped = np.squeeze(y_test)
    pearsonr = stats.pearsonr(y_pred_reshaped, y_test_reshaped)

    # print the best models performance metrics
    print('Test MAE: %.3f' % mae)
    print('Test R2: %.3f' % r2)
    print('Test Pearsonr: %.3f' % pearsonr[0])

    # scatter plot for test set predictions vs. actual values
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred)
    plt.title(f'Predictions vs. Actual\nMAE={mae:.2f}, R2={r2:.2f}, Pearsonr={pearsonr[0]:.2f}')
    plt.ylabel('Actual')
    plt.xlabel('Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

    return best_model
