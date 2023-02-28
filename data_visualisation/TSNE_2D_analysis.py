from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# TODO: plot 2d abYsis bert data

def plot2d(data, dims=2, perp=100):
    light, heavy, name, source = dp.data_extract_abY('../data/abYsis_data.csv')

    X = data
    X_embedded = TSNE(n_components=dims, learning_rate='auto',
                           init = 'random', perplexity = perp).fit_transform(X)

    dataset = pd.DataFrame({'Column1': X_embedded[:, 0], 'Column2': X_embedded[:, 1]})


    dataset['source']= name


    tsne_plot = sns.lmplot(x="Column1", y="Column2", data=dataset, fit_reg=False, hue='source', legend=False,scatter_kws={"s": 10,'alpha': 0.5})
    plt.legend(loc='lower right')
    plt.show()

    tsne_plot.figure.savefig("TSNE_plot_per50.png", bbox_inches='tight')