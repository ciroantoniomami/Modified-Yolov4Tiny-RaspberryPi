import pandas as pd
import numpy as np

df = pd.read_csv('test.csv')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.96

train = df[msk]
test = df[~msk]

test.to_csv('small_test.csv')
