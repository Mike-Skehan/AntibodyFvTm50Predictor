import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import stats
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt


def gradient_boosting_regression(X, y):
    """Gradient Boosting for regression

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        The training input samples.
    y : array-like, shape = [n_samples]
        The target values (real numbers in regression).

    Returns
    -------
    reg : regression model.
    """
    params = {
        "n_estimators": 5000,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
    }
    sns.set_style(style='white')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=13
    )

    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)

    mse = mean_squared_error(y_test, reg.predict(X_test))
    mae = mean_absolute_error(y_test, reg.predict(X_test))
    pearsonr = stats.pearsonr(y_test, reg.predict(X_test))
    r2 = r2_score(y_test, reg.predict(X_test))

    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    print("The mean absolute error (MAE) on test set: {:.4f}".format(mae))
    print("The pearson coeffieicent on test set: {:.4f}".format(pearsonr[0]))
    print("The r2 on test set: {:.4f}".format(r2))

    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = mean_absolute_error(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        reg.train_score_,
        "b-",
        label="Training Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()

    # plot actual vs predicted
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.scatter(reg.predict(X_test),y_test, alpha=0.5)
    plt.plot([60,80], [60,80], 'r--', lw=2)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f'Predictions vs. Actual\nMAE={mae:.2f}, R2={r2:.2f}, Pearsonr={pearsonr[0]:.2f}')
    fig.tight_layout()
    plt.show()

    return reg







