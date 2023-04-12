import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

data = '../data/Jain_Ab_dataset.csv'

def jain_explore(data):
    df = pd.read_csv(data)
    df = df.iloc[:, 0:5]
    df.rename(columns={'LC Class': 'lc_class'}, inplace=True)
    df.rename(columns={'Fab Tm by DSF (°C)': 'tm50'}, inplace=True)
    jain_plot = sns.scatterplot(data=df, x="tm50", y=df.index, hue='lc_class')
    sns.despine(offset=10, trim=True)
    fig = jain_plot.get_figure()
    fig.savefig("Jain_tm50_lc_vis.png", bbox_inches='tight')
    plt.clf()
    classes = df.groupby('lc_class').size()
    classes_sorted = classes.sort_values(ascending=False)
    classes_plot = sns.barplot(x=classes_sorted.values, y=classes_sorted.index)
    sns.despine(offset=10, trim=True)
    fig = classes_plot.get_figure()
    fig.savefig("Jain_classes_vis.png", bbox_inches='tight')
