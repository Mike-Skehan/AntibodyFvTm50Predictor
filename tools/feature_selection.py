from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


def feat_select(X, y, k):
    X_new = SelectKBest(f_regression, k=k).fit_transform(X, y)

    return X_new
