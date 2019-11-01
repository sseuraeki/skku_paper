import pandas as pd
import sys

def label(x, q1, q2, q3, q4):
	if x <= q1:
		return 0
	if x <= q2:
		return 1
	if x <= q3:
		return 2
	if x <= q4:
		return 3
	return 4

if len(sys.argv) != 2:
	print('Usg: python {} csvfile'.format(sys.argv[0]))
	exit()

df = pd.read_csv(sys.argv[1])

if 'roas' in df.columns:
	print(df['roas'].describe(), file=sys.stderr)

	q1 = df['roas'].quantile(0.2) // 0.01 * 0.01
	q2 = df['roas'].quantile(0.4) // 0.01 * 0.01
	q3 = df['roas'].quantile(0.6) // 0.01 * 0.01
	q4 = df['roas'].quantile(0.8) // 0.01 * 0.01

	df['roas'] = df['roas'].apply(lambda x: label(x, q1, q2, q3, q4))

# remove unneeded cols
del df['date'], df['appid'], df['campaign']
df.to_csv(sys.stdout, index=False)



