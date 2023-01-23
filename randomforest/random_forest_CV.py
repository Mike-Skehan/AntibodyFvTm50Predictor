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

cv = 5

regr = RandomForestRegressor(random_state=0)

rf_random = RandomizedSearchCV(estimator=regr,
                               param_distributions=random_grid, n_iter=100,
                               cv=3, verbose=2, n_jobs=-1)

scores = cross_val_score(rf_random, X, Y, scoring='r2', cv=cv, n_jobs=-1)

print(scores)
