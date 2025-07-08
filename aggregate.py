import os

import pandas as pd

files = os.listdir('outputs')
dfs = [pd.read_csv(f'outputs/{file}') for file in files if file.endswith('.csv')]
df = pd.concat(dfs, ignore_index=True)
df.sort_values(['pearson', 'spearman'], inplace=True, ascending=False)
print(df)