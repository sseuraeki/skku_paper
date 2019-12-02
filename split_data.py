import numpy as np
import pandas as pd
import os
import sys
from sklearn.utils import shuffle

if len(sys.argv) != 2:
	print('Usg: python {} csvfile'.format(sys.argv[0]))
	exit()

SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(sys.argv[1])

# split by labels
label_dfs = []
for label in range(5):
	label_dfs.append(df[df['roas']==label])

# split each label df
train_dfs = []
valid_dfs = []
test_dfs = []

for label in range(5):
	label_df = label_dfs[label]
	train_df = label_df.sample(frac=0.8, random_state=1)
	train_dfs.append(train_df)

	label_df = label_df.drop(train_df.index)
	valid_df = label_df.sample(frac=0.5, random_state=1)
	valid_dfs.append(valid_df)

	test_df = label_df.drop(valid_df.index)
	test_dfs.append(test_df)

trainset = pd.concat(train_dfs, ignore_index=True, sort=False)
validset = pd.concat(valid_dfs, ignore_index=True, sort=False)
testset = pd.concat(test_dfs, ignore_index=True, sort=False)

# shuffle data
trainset = shuffle(trainset)
validset = shuffle(validset)
testset = shuffle(testset)

# write
trainset.to_csv('{}/data/trainset.csv'.format(SCRIPTPATH), index=False)
validset.to_csv('{}/data/validset.csv'.format(SCRIPTPATH), index=False)
testset.to_csv('{}/data/testset.csv'.format(SCRIPTPATH), index=False)

