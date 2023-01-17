from sklearn.model_selection import train_test_split
import data_parser as dp
import sys
from sklearn.manifold import TSNE
from scipy import stats
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
import numpy as np

sys.path.insert(0, "../data/")
sys.path.insert(0, "../")
sys.path.insert(0, "../tools")

bert_data = "../data/combined_bert_df.csv"

df = pd.read_csv(bert_data)

light, heavy, temp = dp.data_extract_Jain('../data/combined_datasets.csv')

X = df
Y = temp

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = [1.0, 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

regr = RandomForestRegressor(random_state=0)


def cross_validation(model, _X, _y, _cv=5):
    '''Function to perform 5 Folds Cross-Validation
     Parameters
     ----------
    model: Python Class, default=None
            This is the machine learning algorithm to be used for training.
    _X: array
         This is the matrix of features.
    _y: array
         This is the target variable.
    _cv: int, default=5
        Determines the number of folds for cross-validation.
     Returns
     -------
     The function returns a dictionary containing the metrics 'accuracy', 'precision',
     'recall', 'f1' for both training set and validation set.
    '''
    _scoring = ['accuracy', 'precision', 'recall', 'f1']

    results = cross_validate(estimator=model,
                             X=_X,
                             y=_y,
                             cv=_cv,
                             scoring='accuracy',
                             return_train_score=True)

    return results


decision_tree_result = cross_validation(regr, X, Y, 5)

print(decision_tree_result)
