from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def feat_select(X, y, k):

    X_new = SelectKBest(f_regression, k=k).fit_transform(X, y)

    return X_new


def rfe_select(X, y, n_feats):

    # Create a non-linear model (Random Forest Regressor)
    rf = RandomForestRegressor()

    # Create an RFE object and specify the non-linear model to use
    rfe = RFE(estimator=rf, n_features_to_select=n_feats)

    # Fit the RFE object to the data to select the best features
    rfe.fit(X, y)

    # Get the selected features
    selected_features = X.columns[rfe.support_]

    return selected_features


def rfe_plot(X, y):

    # Create a non-linear model (Random Forest Regressor)
    rf = RandomForestRegressor()

    # Create a range of values for n_features_to_select to try
    n_features_range = range(1, X.shape[1] + 1, 24)

    # Create an empty list to store the cross-validation scores for each value of n_features_to_select
    cv_scores = []

    # Loop over the values of n_features_to_select
    for n_features in n_features_range:
        # Create an RFE object with the non-linear model and the current value of n_features_to_select
        rfe = RFE(estimator=rf, n_features_to_select=n_features)

        # Perform cross-validation and get the mean score across folds
        score = cross_val_score(rfe, X, y, cv=5).mean()

        # Append the score to the list of scores
        cv_scores.append(score)

    # Plot the cross-validation scores as a function of n_features_to_select
    plt.plot(n_features_range, cv_scores)
    plt.xlabel('Number of features selected')
    plt.ylabel('Cross-validation score')
    return plt.show()