import pandas as pd
import os
import sys
import glob
import matplotlib.pyplot as plt

SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
df = pd.concat(
	[pd.read_csv(f) for f in glob.glob('{}/joined/labeled/*/joined*.csv'.format(SCRIPTPATH))],
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


# describe labels
print(df['roas'].describe(), file=sys.stderr)

# write
df.to_csv(sys.stdout, index=False)