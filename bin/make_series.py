import numpy as np
import pandas as pd
import sys

if len(sys.argv) != 3:
	print('Usg: python {} csvfile outfile'.format(sys.argv[0]))
	exit()

df = pd.read_csv(sys.argv[1])

series_cols = ['retention', 'return', 'dau', 'mau', 'arpdau', 'act_sess', 'cpi', 'cpc']
intervals = [1, 2, 3, 7]

result = []

for i in range(len(df)):
	tmp = df.iloc[i].copy()
	
	row = np.zeros((len(intervals), len(series_cols)))
	for c in range(len(series_cols)):
		target_cols = ['{}_d{}'.format(series_cols[c], interval) for interval in intervals]
		series = tmp[target_cols].values
		row[:, c] = series

	result.append(row)

result = np.array(result)

# write
np.save(sys.argv[2], result)


