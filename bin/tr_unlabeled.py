# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import warnings
import pandas as pd

# assertions ------
warnings.filterwarnings('ignore')

if len(sys.argv) != 6:
	print('Usg: python {} base.csv retention.csv return.csv activity.csv activity2.csv'.format(sys.argv[0]))
	exit()
# ------

base = pd.read_csv(sys.argv[1])
retention = pd.read_csv(sys.argv[2])
return_ = pd.read_csv(sys.argv[3])
activity = pd.read_csv(sys.argv[4])
activity2 = pd.read_csv(sys.argv[5])

# join base & retention
result = pd.merge(base, retention, on=['appid', 'date'], how='left')

# join return
return_['campaign'] = return_['c1']
del return_['c1']
result = pd.merge(result, return_, on=['campaign', 'date'], how='left')

# join activity
result = pd.merge(result, activity, on=['appid'], how='left')

# join activity2
result = pd.merge(result, activity2, on=['appid'], how='left')

result.to_csv(sys.stdout, index=False)
