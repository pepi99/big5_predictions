import pandas as pd
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

df = pd.read_csv('../twitter_data.csv')[16378:]
column_names_df = pd.read_csv('../column_names.csv', header=None)
column_names = column_names_df.iloc[:, 3].array.to_numpy()
df.columns = column_names

df = df[['id', 'big5_openness']]
actual = df['big5_openness'].to_numpy()

cols = ['big5_openness_actual']
idx = df['id'].to_numpy()

new_df = pd.DataFrame(data=actual, columns=cols, index=idx)
new_df.to_csv('../results.csv')
