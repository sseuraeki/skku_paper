import numpy as np
import pandas as pd
import os
import sys
import pickle

# assertions
if (len(sys.argv) < 2) or (len(sys.argv) > 3):
	print('Usg: python {} csvfile normalize_factors(optional)'.format(sys.argv[0]))
	exit()

plain_cols = ['spend', 'install']
series_cols = ['retention', 'return', 'dau', 'mau', 'arpdau', 'act_sess', 'cpi', 'cpc']
intervals = [1, 2, 3, 7]
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(sys.argv[1])

# load normalize factors
normalize_factors = {}
if len(sys.argv) == 3:
	with open(sys.argv[2], 'rb') as f:
		normalize_factors = pickle.load(f)

# normalize plain cols
for col in plain_cols + series_cols:
	if col in plain_cols:
		target_cols = [col]
	else:
		target_cols = ['{}_d{}'.format(col, i) for i in intervals]

	if len(sys.argv) == 3:
		min_val = normalize_factors[col][0]
		max_val = normalize_factors[col][1]
	else:
		min_val = df[target_cols].values.min()
		max_val = df[target_cols].values.max()
		normalize_factors[col] = (min_val, max_val)

	for col in target_cols:
		df[col] = df[col].apply(
			lambda x: 2 * ((x - min_val) / (max_val - min_val)) - 1.0)  # -1 ~ 1

# write normalize factors
if len(sys.argv) == 2:  # if factors not given
	with open('{}/normalize_factors.pkl'.format(SCRIPTPATH), 'wb') as f:
		pickle.dump(normalize_factors, f)

df.to_csv(sys.stdout, index=False)






