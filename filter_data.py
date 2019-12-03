import pandas as pd
import os
import sys
import glob
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
	print('Usg: python {} labeled/unlabeled'.format(sys.argv[0]))
	exit()

if sys.argv[1] not in ['labeled', 'unlabeled']:
	print('Usg: python {} labeled/unlabeled'.format(sys.argv[0]))
	exit()

SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
df = pd.concat(
	[pd.read_csv(f) for f in glob.glob('{}/joined/{}/*/joined*.csv'.format(SCRIPTPATH, sys.argv[1]))],
	ignore_index=True,
	sort=False)

# total records
print('Total records: {}'.format(len(df)), file=sys.stderr)

# remove null records
df = df.dropna()
df = df.drop_duplicates()
print('Null dropped: {}'.format(len(df)), file=sys.stderr)

# remove records with no spend
df = df[df['spend'] >= 10.0]
print('No spend dropped: {}'.format(len(df)), file=sys.stderr)

# remove records with abnormal retention
df = df[df['retention_d1'] <= 100.0]
df = df[df['retention_d2'] <= 100.0]
df = df[df['retention_d3'] <= 100.0]
df = df[df['retention_d7'] <= 100.0]
print('Abnormal retention dropped: {}'.format(len(df)), file=sys.stderr)

# return > ratio
df['return_d1'] = df['return_d1'] / df['spend']
df['return_d2'] = df['return_d2'] / df['spend']
df['return_d3'] = df['return_d3'] / df['spend']
df['return_d7'] = df['return_d7'] / df['spend']

# drop abnormal returns
df = df[df['return_d1'] < 1.0]
df = df[df['return_d2'] < 1.0]
df = df[df['return_d3'] < 1.0]
df = df[df['return_d7'] < 1.0]
print('Abnormal return dropped: {}'.format(len(df)), file=sys.stderr)

# write
df.to_csv(sys.stdout, index=False)