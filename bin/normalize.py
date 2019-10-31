import numpy as np
import pandas as pd
import os
import sys
import glob
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1])

plain_cols = ['spend', 'install']
series_cols = ['retention', 'return', 'dau', 'mau', 'arpdau', 'act_sess', 'cpi', 'cpc']
intervals = [1, 2, 3, 7]

normalize_factors = {}

# normalize plain cols
for plain_col in plain_cols:
	min_val = df[plain_col].min()
	max_val = df[plain_col].max()

	normalize_factors[plain_col] = (min_val, max_val)

	df[plain_col] = df[plain_col].apply(
		lambda x: 2 * ((x - min_val) / (max_val - min_val)) - 1.0)  # -1 ~ 1

# normalize series cols
for series_col in series_cols:
	cols = ['{}_d{}'.format(series_col, i) for i in intervals]
	min_val = df[cols].values.min()
	max_val = df[cols].values.max()

	normalize_factors[series_col] = (min_val, max_val)

	for col in cols:
		df[col] = df[col].apply(
			lambda x: 2 * ((x - min_val) / (max_val - min_val)) - 1.0)

print(normalize_factors, file=sys.stderr)
df.to_csv(sys.stdout, index=False)






