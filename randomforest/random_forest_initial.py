import numpy as np
import sys
from time import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from tools import data_parser as dp
import tools.encoding as encoding
from sklearn.manifold import TSNE
from scipy import stats

sys.path.insert(0,"../data/")
sys.path.insert(0,"../")
sys.path.insert(0,"../tools")

if __name__ == '__main__':

    light, heavy, temp = dp.data_extract('../data/Jain_Ab_dataset.csv')

    concat_seq = encoding.concat_seq(light, heavy)
    Y = temp

    protvec_array = encoding.seq2vec(concat_seq)
    X = TSNE(n_components=50, learning_rate='auto', init='random', perplexity=30,
                      method='exact').fit_transform(protvec_array)

# Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
    max_features = [1.0, 'sqrt']
# Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
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

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    search_start = time()

    regr = RandomForestRegressor()

    rf_random = RandomizedSearchCV(estimator=regr,
                                   param_distributions=random_grid, n_iter=100,
                                   cv=3, verbose=2, random_state=42, n_jobs=-1)

    search_end = time()

    search_duration = search_end-search_start

    rf_random.fit(x_train, y_train)

    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        r2 = stats.pearsonr(predictions, test_labels)
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Pearson Coeff: {:0.2f}'.format(r2[0]))

        return r2[0]

    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, x_test, y_test)

    print (random_accuracy)
