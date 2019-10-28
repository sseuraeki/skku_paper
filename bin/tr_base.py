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
	print('Usg: python {} base.csv'.format(sys.argv[0]))
	exit()
# ------

# transform base ------
base = pd.read_csv(sys.argv[1])

base['campaign'] = base['c1']
base['date'] = base['from_']

spend = base[base['kpi'] == 'campaign.all.spend.value'].copy()
install = base[base['kpi'] == 'campaign.all.install.value'].copy()

cols = ['date', 'appid', 'campaign']
spend['spend'] = spend['d1']
install['install'] = install['d1']

spend = spend[cols + ['spend']]
install = install[cols + ['install']]

result = pd.merge(spend, install, on=cols, how='left')
result.to_csv(sys.stdout, index=False)
# ------
