from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# TODO: plot 2d abYsis bert data


def plot2d(data, source, dims=2, perpx=100):

    df = pd.read_csv(data)
    Array2d = df.to_numpy()

    X_embedded = TSNE(n_components=dims, learning_rate='auto',
                           init ='random', perplexity=perpx).fit_transform(Array2d)

    dataset = pd.DataFrame({'Column1': X_embedded[:, 0], 'Column2': X_embedded[:, 1]})

    dataset['source'] = source

    tsne_plot = sns.lmplot(x="Column1", y="Column2", data=dataset, fit_reg=False, hue='source', legend=False,
                           scatter_kws={"s": 10,'alpha': 0.5})
    plt.legend(loc='lower right')
    plt.show()

    return tsne_plot.figure.savefig("TSNE_plot_per50.png", bbox_inches='tight')