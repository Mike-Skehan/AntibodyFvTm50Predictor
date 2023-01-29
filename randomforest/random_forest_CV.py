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

    rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=params, n_iter=iters,
                                   cv=cv_num, verbose=2, n_jobs=-1)

    rf_random.fit(x_train, y_train)

    return rf_random.best_estimator_


def eval_model(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = mean_absolute_error(test_labels, predictions)
    r2 = stats.pearsonr(predictions, test_labels)
    print('Model Performance\n---------------------')
    print('Average Error: {:0.2f} degrees.'.format(np.mean(errors)))

    return 'Pearson Coeff: {:0.2f}'.format(r2[0])
    return np.mean(errors), r2[0] 


def eval_avg(test_labels):
    avg_temp = np.mean(test_labels)
    avg_list =[]
    for i in range(len(test_labels)):
        avg_list.append(avg_temp)
    errors = mean_absolute_error(test_labels, avg_list)
    return np.mean(errors)


def compare_to_avg(model, test_features, test_labels):
        model_mae   = eval_model(model, test_features, test_labels)
        avg_mae     = eval_avg(test_labels)
        improvement = ((avg_mae-model_mae[0])/avg_mae)*100
    
        return 'Model error  : {:0.2f} C\nAverage error: {:0.2f} C\nImprovement  : {:0.2f}%'.format(model_mae[0],
                                                                                                    avg_mae,
                                                                                                    improvement)
    

def save_result(model, dataset, model_name, model_loc, pearson_result):
    try:
        df   = pd.read_csv('results.csv')
    except FileNotFoundError:
        df   = pd.DataFrame(columns=['Dataset', 'Model', 'Model File', 'Features File', 'MAE', 'Pearson Coeff'])
    joblib.dump(model, model_loc)
    new_data = {
        'Dataset': [dataset],
        'Model': [model_name],
        'Model File': [model_loc],
        'Pearson': [pearson_result]
    }

    df = df.append(new_data)

    df.to_csv('./models/results.csv', index=False)
    


    


