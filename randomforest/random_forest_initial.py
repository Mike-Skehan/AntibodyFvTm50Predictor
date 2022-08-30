import pandas as pd
import numpy as np
import sys
from time import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split


sys.path.insert(0,"../data/")
sys.path.insert(0,"../")

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

def data_extract_Jain(data_file):
    df = pd.read_csv(data_file)
    df.drop([0, 4])
    df.rename(columns={'VL': 'Light'}, inplace=True)
    df.rename(columns={'VH': 'Heavy'}, inplace=True)
    df.rename(columns={"Fab Tm by DSF (°C)": 'Temp'}, inplace=True)

    light_seq = df['Light'].values.tolist()
    heavy_seq = df['Heavy'].values.tolist()
    temp = df['Temp'].values.tolist()

    # l_seq_list = remove_special_chars(light_seq)
    # h_seq_list = remove_special_chars(heavy_seq)

    return light_seq, heavy_seq, temp

def one_hot_encoder(sequences, max_length):

    """
    General info
    :param sequences    : list of Fv amino acid sequences.
    :param max_length   : maximum sequence length of Fv amino acid sequence.
    :return             : amino acid sequences encoded in the one hot form.
    """
    one_hot_seq = np.zeros((len(sequences), max_length, len(amino_acids)), dtype='int')

    for x, seq in enumerate(sequences):
        for y, aa in enumerate(seq):
            loc = amino_acids.find(aa)
            if loc > 0:
                    one_hot_seq[x, y, loc] = 1
    return one_hot_seq

light, heavy, temp = data_extract_Jain('../data/Jain_Ab_dataset.csv')

concated = [''.join(z) for z in zip(heavy, light)]

encoded = one_hot_encoder(concated, 250)
newarray = encoded.reshape((np.shape(encoded)[0]),(np.shape(encoded)[1]* np.shape(encoded)[2]))
print(np.shape(newarray))

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
               'max_features': [1, 'sqrt'],
               'max_depth': max_depth,
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}

x_train, x_test, y_train, y_test = train_test_split(newarray, temp)

search_start = time()

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
rf_random.fit(x_train, y_train)

search_end = time()

search_duration = search_end-search_start

print('Best params:', rf_random.best_params_)
print('Best score: ', f'{rf_random.best_score_:4.3f}')
print(f'Search time: {search_duration:3.1f} secs')
best_params = rf_random.best_params_

randmf = RandomForestRegressor(**best_params)
randmf.fit(x_train,y_train)

y_pred_rf1 = pd.DataFrame( { "actual": y_test,
"predicted_prob": randmf.predict((x_test)),'score': randmf.score(x_test,y_test)})

print(y_pred_rf1)

