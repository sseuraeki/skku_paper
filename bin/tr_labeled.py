# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import warnings
import pandas as pd

# assertions ------
warnings.filterwarnings('ignore')

if len(sys.argv) != 8:
	print(
		'Usg: python {} base.csv retention.csv return.csv activity.csv activity2.csv iap.csv adrev.csv'.format(
			sys.argv[0]
			)
		)
	exit()
# ------

base = pd.read_csv(sys.argv[1])
retention = pd.read_csv(sys.argv[2])
return_ = pd.read_csv(sys.argv[3])
activity = pd.read_csv(sys.argv[4])
activity2 = pd.read_csv(sys.argv[5])
iap = pd.read_csv(sys.argv[6])
adrev = pd.read_csv(sys.argv[7])

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

# join iap
iap = iap[['campaign', 'iap']]
result = pd.merge(result, iap, on='campaign', how='left')

# join adrev
result = pd.merge(result, adrev, on='campaign', how='left')

# calc roas
result['iap'] = result['iap'].apply(lambda x: float(x))
result['adrev'] = result['adrev'].apply(lambda x: float(x))
result['spend'] = result['spend'].apply(lambda x: float(x))

result['rev'] = result['iap'] + result['adrev']
result['roas'] = result['rev'] / result['spend']
del result['iap'], result['adrev'], result['rev']

result.to_csv(sys.stdout, index=False)
