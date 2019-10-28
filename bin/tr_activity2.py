# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import warnings
import pandas as pd

def convert(df, metric):
	cols = []
	for i in [1, 2, 3, 7]:
		oldcol = 'd{}'.format(i)
		newcol = '{}_d{}'.format(metric, i)
		df[newcol] = df[oldcol]
		cols.append(newcol)
	return df[['appid'] + cols]

# assertions ------
warnings.filterwarnings('ignore')

if len(sys.argv) != 2:
	print('Usg: python {} csv'.format(sys.argv[0]))
	exit()
# ------

dataset = pd.read_csv(sys.argv[1])

cpi = dataset[dataset['kpi'] == 'adnetads.all.cpi.value'].copy()
cpc = dataset[dataset['kpi'] == 'adnetpub.all.cpc.value'].copy()

cpi = convert(cpi, 'cpi')
cpc = convert(cpc, 'cpc')

result = pd.merge(cpi, cpc, on='appid', how='left')
result.to_csv(sys.stdout, index=False)
