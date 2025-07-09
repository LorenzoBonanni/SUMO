import os

import pandas as pd

files = os.listdir('outputs')
dfs = [pd.read_csv(f'outputs/{file}') for file in files if file.endswith('.csv')]
df = pd.concat(dfs, ignore_index=True)
grouped = df.groupby(['uncertainty', 'dataset']).mean().reset_index()
grouped = grouped[['uncertainty', 'dataset', 'pearson', 'spearman']]
spearman_table = grouped.pivot(index='dataset', columns='uncertainty', values='spearman')
pearson_table = grouped.pivot(index='dataset', columns='uncertainty', values='pearson')
spearman_table.loc['average'] = spearman_table.mean()
pearson_table.loc['average'] = pearson_table.mean()
spearman_table.to_csv('outputs/spearman.csv')
pearson_table.to_csv('outputs/pearson.csv')