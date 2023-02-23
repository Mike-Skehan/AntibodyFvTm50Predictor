import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

# TODO: write functions based on code.

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10)

data = '../data/abYsis_data.csv'

df = pd.read_csv(data)

s = df.groupby('organism').size().reset_index(name='total')
s_sorted = s.sort_values('total', ascending=False)

print(s_sorted)

s_sorted['percent'] = (s_sorted['total']/s_sorted['total'].sum())*100

print(s_sorted)

# abYsis_plot = sns.barplot(x=s_sorted.values[:10],y=s_sorted.index[:10])
# sns.despine(offset=10, trim=True)

# fig = abYsis_plot.get_figure()
# fig.savefig("abYsis_organism_vis.png", bbox_inches='tight')


df2 = df[(df.organism == 'mus musculus') | (df.organism == 'homo sapiens')]

df2["heavy_length"] = df2["heavy"].str.len()
df2["light_length"] = df2["light"].str.len()

# print(df2['heavy_length'].max())
# plt.figure()

sns.distplot(df2['heavy_length'], hist=False, kde_kws=dict(alpha=0.5))
sns.distplot(df2['light_length'], hist=False, kde_kws=dict(alpha=0.5))
sns.despine()
plt.axvline(80, 0, 1, alpha=0.5, linestyle='--', color='green')
plt.axvline(150, 0, 1, alpha=0.5, ls='--', color='green')
plt.xlabel('sequence length')
plt.legend(labels=["Heavy", "Light"])

plt.show()

df3 = df2[(df2.heavy_length <= 150) & (df2.heavy_length >= 80) & (df2.light_length <= 150) & (df2.light_length >= 80)]

print(df3.describe())

sns.distplot(df3['heavy_length'], hist=False, kde_kws=dict(alpha=0.5))
sns.distplot(df3['light_length'], hist=False, kde_kws=dict(alpha=0.5))
sns.despine()

plt.xlabel('sequence length')
plt.legend(labels=["Heavy", "Light"])
plt.show()
