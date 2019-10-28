# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import warnings
import pandas as pd

# assertions ------
warnings.filterwarnings('ignore')

if len(sys.argv) != 2:
	print('Usg: python {} csv'.format(sys.argv[0]))
	exit()
# ------

dataset = pd.read_csv(sys.argv[1])

# find grouping cols
if 'c1' in dataset.columns:
	grouping = ['from_', 'c1']
else:
	grouping = ['appid', 'from_', 'geo', 'pid']

# split by kpis
kpis = dataset['kpi'].unique()

split = dataset[dataset['kpi'] == kpis[0]][grouping + ['d1', 'kpi']].copy()
new_col = '_'.join(kpis[0].split('.')[-2:])
split[new_col] = split['d1']
del split['d1'], split['kpi']

for kpi in kpis[1:]:
	tmp = dataset[dataset['kpi'] == kpi][grouping + ['d1', 'kpi']].copy()
	new_col = '_'.join(kpi.split('.')[-2:])
	tmp[new_col] = tmp['d1']
	del tmp['d1'], tmp['kpi']
	split = pd.merge(split, tmp, on=grouping, how='left')

# remove unnecessary cols
if 'c1' not in dataset.columns:
	appids = split['appid'].unique()
	geos = split['geo'].unique()
	pids = split['pid'].unique()

	if (len(appids) == 1) and (appids[0] in ['all', 'filtered']):
		del split['appid']
	if (len(geos) == 1) and (geos[0] in ['all', 'filtered']):
		del split['geo']
	if (len(pids) == 1) and (pids[0] in ['all', 'filtered']):
		del split['pid']

split['date'] = split['from_']
del split['from_']

split.to_csv(sys.stdout, index=False)

