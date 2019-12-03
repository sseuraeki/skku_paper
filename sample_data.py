import pandas as pd
import sys
from sklearn.utils import shuffle

if len(sys.argv) != 4:
	print('Usg: python {} csvfile n_labels n_samples'.format(sys.argv[0]))
	exit()

n_labels = int(sys.argv[2])
n_samples = int(sys.argv[3])

df = pd.read_csv(sys.argv[1])

samples = []
for label in range(n_labels):
	tmp = df[df['roas'] == label]
	samples.append(tmp.sample(n=n_samples, random_state=1))

sampled = pd.concat(samples, ignore_index=True, sort=False)
sampled = shuffle(sampled)

sampled.to_csv(sys.stdout, index=False)
