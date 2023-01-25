import matplotlib.pyplot as plt


def rf_plot(x_test, y_test, model):
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, model.predict(x_test), c='crimson')
    p1 = max(max(model.predict(x_test)), max(y_test))
    p2 = min(min(model.predict(x_test)), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()
