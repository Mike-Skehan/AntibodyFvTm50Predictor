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

sys.path.insert(0, "../data/")
sys.path.insert(0, "../")
sys.path.insert(0, "../tools")


def grid_search(x_train, y_train, params, iters, cv_num):

    rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=params, n_iter=iters,
                                   cv=cv_num, verbose=2, n_jobs=-1)

    rf_random.fit(x_train, y_train)

    return rf_random.best_estimator_
