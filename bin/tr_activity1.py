# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import warnings
import pandas as pd
import datetime as dt

# functions ------
def get_appid(af_appid):
	try:
		if af_appid[:2] == 'id':
			tmp = liveapps[liveapps['storekey'] == af_appid[2:]]
			return tmp['appid'].values[0]
		return af_appid
	except:
		return ''
# ------

# assertions ------
warnings.filterwarnings('ignore')

if len(sys.argv) != 2:
	print('Usg: python {} activity.split.csv'.format(sys.argv[0]))
	exit()
# ------

# read data ------
data = pd.read_csv(sys.argv[1])
# ------

# get dates ------
dates = data['Install Time'].unique().tolist()
min_date = min(dates)
# ------

# transform ------
result = pd.DataFrame(data['App ID'].unique(), columns=['appid'])

for i in [0, 1, 2, 6]:
	date = dt.datetime.strftime(dt.datetime.strptime(min_date, '%Y-%m-%d') + dt.timedelta(i), '%Y-%m-%d')
	tmp = data[data['Install Time'] == date].copy()
	cols = ['appid', 'date',
	        'dau_d{}'.format(i+1), 'mau_d{}'.format(i+1),
	        'arpdau_d{}'.format(i+1), 'act_sess_d{}'.format(i+1)]
	tmp.columns = cols
	del tmp['date']
	result = pd.merge(result, tmp, on='appid', how='left')

url = 'http://roimon.datawave.co.kr/api/v3/apps?islive=1'
liveapps = pd.read_json(url, orient='records')
result['appid'] = result['appid'].apply(get_appid)

result.to_csv(sys.stdout, index=False)






