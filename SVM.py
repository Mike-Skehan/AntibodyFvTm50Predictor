import numpy as np
from scipy.stats import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
import statistics as st
import pandas as pd


def svm_cv(X, y):

    """

    :param X: Features
    :param y: Labels
    :return: Best model
    """

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

    for i, (train_index, val_index) in enumerate(kf.split(X_train)):

        k_X_train, X_val    = X_train.iloc[train_index], X_train.iloc[val_index]
        k_y_train, y_val    = y_train[train_index], y_train[val_index]

        # create the SVM model
        model = SVR(kernel='rbf')

        # define the parameter grid to search over
        param_dist      = {'C': uniform(loc=0, scale=1000), 'gamma': ['scale', 'auto'] + list(np.logspace(-5, 2, 10))}

        # set up the randomized search
        random_search   = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=kf, n_jobs=-1)

        random_search.fit(k_X_train, k_y_train)

        # evaluate the model on the test set
        predictions     = random_search.predict(X_val)
        mae             = mean_absolute_error(y_val, predictions)
        r2              = r2_score(y_val, predictions)

        # y_test and predictions are arrays with shape (36, 1)
        y_val_reshaped          = np.squeeze(y_val)
        predictions_reshaped    = np.squeeze(predictions)

        # Calculate Pearson correlation coefficient
        pearsonr        = stats.pearsonr(y_val_reshaped, predictions_reshaped)

        # keep track of the best model based on validation score
        if r2 > best_score:
            best_score  = r2
            best_model  = model

        mae_scores.append(mae)
        r2_scores.append(r2)
        pearsonr_scores.append(pearsonr[0])

        # scatter plot predictions vs. actual values on the appropriate subplot
        axs[i].scatter(y_val, predictions)
        axs[i].set_title(f'Fold {i+1} Predictions vs. Actual\nMAE={mae:.2f}, R2={r2:.2f}, Pearsonr={pearsonr[0]:.2f}')
        axs[i].set_ylabel('Actual')
        axs[i].set_xlabel('Predictions')
        axs[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

    # print the model's mean and standard dev performance metrics
    print('Validation MAE: %.3f (%.3f)' % ((st.mean(mae_scores)), st.stdev(mae_scores)))
    print('Validation R2: %.3f (%.3f)' % ((st.mean(r2_scores)), st.stdev(r2_scores)))
    print('Validation Pearsonr: %.3f (%.3f)' % ((st.mean(pearsonr_scores)), (st.stdev(pearsonr_scores))))

    best_model.fit(X_train, y_train)

    y_pred              = best_model.predict(X_test)
    mae                 = mean_absolute_error(y_test, y_pred)
    r2                  = r2_score(y_test, y_pred)

    y_pred_reshaped     = np.squeeze(y_pred)
    y_test_reshaped     = np.squeeze(y_test)
    pearsonr            = stats.pearsonr(y_pred_reshaped, y_test_reshaped)

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

