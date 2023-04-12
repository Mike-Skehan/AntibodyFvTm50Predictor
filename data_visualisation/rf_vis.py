import matplotlib.pyplot as plt
import numpy as np


def rf_plot(x_test, y_test, model):
    """

    :param x_test: test features
    :param y_test: test labels
    :param model: random forest regression model
    :return:
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, model.predict(x_test), c='crimson')
    p1 = max(max(model.predict(x_test)), max(y_test))
    p2 = min(min(model.predict(x_test)), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.figure.savefig()


def avg_plot(y_test):
    plt.figure(figsize=(10, 10))
    avg_temp = np.mean(y_test)
    avg_list = []
    for i in range(len(y_test)):
        avg_list.append(avg_temp)
    plt.scatter(y_test, avg_list, c='crimson')
    p1 = max(max(avg_list), max(y_test))
    p2 = min(min(avg_list), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()
