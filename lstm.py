import numpy as np
import pandas as pd
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from tensorflow.keras.utils import to_categorical
from keras.metrics import accuracy

train_x = np.load('./data/train_series.npy')
test_x = np.load('./data/test_series.npy')

train_y = to_categorical(pd.read_csv('./data/trainset.csv')['roas'].values)
test_y = to_categorical(pd.read_csv('./data/testset.csv')['roas'].values)

def build_lstm(seq_shape, n_classes, hidden_units=32):
	seq_layer = Input(shape=(seq_shape[0], seq_shape[1],))

	h = LSTM(hidden_units)(seq_layer)

	y = Dense(n_classes, activation='softmax')(h)

	model = Model(seq_layer, y)
	model.compile(loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['acc'])

	return model

seq_shape = train_x.shape[1:]

lstm = build_lstm(seq_shape, 5)
lstm.fit(train_x,
	train_y,
	batch_size=128,
	epochs=10)

# predict
predictions = lstm.predict(test_x)

# acc
correct = 0
for i in range(test_x.shape[0]):
	if np.argmax(predictions[i]) == np.argmax(test_x[i]):
		correct += 1

print(correct / len(test_x))