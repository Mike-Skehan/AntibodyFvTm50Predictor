from sklearn.model_selection import train_test_split
import data_parser as dp
import sys
from sklearn.manifold import TSNE
from scipy import stats
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import time
from sklearn.metrics import mean_absolute_error
from statistics import mean
import joblib

sys.path.insert(0, "../data/")
sys.path.insert(0, "../")
sys.path.insert(0, "../tools")


def grid_search(x_train, y_train, params, iters, cv_num):

    rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=params, n_iter=iters,
                                   cv=cv_num, verbose=2, n_jobs=-1)

    rf_random.fit(x_train, y_train)

    return rf_random.best_estimator_


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = mean_absolute_error(test_labels, predictions)
    r2 = stats.pearsonr(predictions, test_labels)
    print('Model Performance\n---------------------')
    print('Average Error: {:0.2f} degrees.'.format(np.mean(errors)))

    return 'Pearson Coeff: {:0.2f}'.format(r2[0])


def save_result(model, dataset, model_name, model_loc, pearson_result):
    try:
        df = pd.read_csv('results.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Dataset', 'Model', 'Model File', 'Features File', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
    joblib.dump(model, './models/250123_RF.joblib')
    new_data = {
        'Dataset': [dataset],
        'Model': [model_name],
        'Model File': [model_loc],
        'Pearson': [pearson_result]
    }

    df = df.append(new_data)

    df.to_csv('./models/results.csv', index=False)

