import sys
import numpy as np
import pandas as pd
from keras.models import model_from_json
from tensorflow.keras.utils import to_categorical

if len(sys.argv) != 3:
	print('python {} model_json model_weights'.format(sys.argv[0]))
	exit()

# load model
json_file = open(sys.argv[1], 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# load weights
model.load_weights(sys.argv[2])

# compile and test
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
X = np.load('data/test_series.npy')
Y = to_categorical(pd.read_csv('./data/testset.csv')['roas'].values)
score = model.evaluate(X, Y)
print('Loss:', score[0])
print('accuracy:', score[1])

