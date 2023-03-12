import sys
from scipy import stats
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import joblib

sys.path.insert(0, "../data/")
sys.path.insert(0, "../")
sys.path.insert(0, "../tools")


def grid_search(x_train, y_train, params, iters, cv_num):
    """

    :param x_train  : training features
    :param y_train  : training labels
    :param params   : grid search parameters
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
    :return          : model error and pearson coefficient
    """

    predictions = model.predict(x_test)
    errors = mean_absolute_error(y_test, predictions)
    r2 = stats.pearsonr(predictions, y_test)
    #('Model Performance\n---------------------\n')
    #print('Average Error: {:0.2f} degrees.'.format(np.mean(errors)))

    return np.mean(errors), r2[0] 


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
    model_mae   = eval_model(model, x_test, y_test)
    avg_mae     = eval_avg(y_test)
    improvement = ((avg_mae-model_mae[0])/avg_mae)*100
    
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
        df   = pd.read_csv('../models/results.csv')
    except FileNotFoundError:
        df   = pd.DataFrame(columns=['Dataset', 'Model', 'Model File', 'Features File', 'MAE', 'Pearson Coeff'])
    joblib.dump(model, model_loc)
    new_data = {
        'Dataset': [dataset],
        'Model': [model_name],
        'Model File': [model_loc],
        'Pearson': [pearson_result]
    }

    df = df.append(new_data,ignore_index=True)

    df.to_csv('../models/results.csv', index=False)
