import pandas as pd
import sys

def label(x, quantiles):
	n_classes = len(quantiles)
	for n_class in range(n_classes):
		if x <= quantiles[n_class]:
			return n_class
	return len(quantiles)

if len(sys.argv) != 3:
	print('Usg: python {} csvfile n_classes'.format(sys.argv[0]))
	exit()

df = pd.read_csv(sys.argv[1])
n_classes = int(sys.argv[2])

quantiles = []
for n_class in range(n_classes):
	quantile = 1.0 / n_classes * (n_class + 1)
	quantile = df['roas'].quantile(quantile) // 0.01 * 0.01
	quantiles.append(quantile)

df['roas'] = df['roas'].apply(lambda x: label(x, quantiles))

# remove unneeded cols
del df['date'], df['appid'], df['campaign']
df.to_csv(sys.stdout, index=False)



