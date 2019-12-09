import sys
import numpy as np
import pandas as pd
from keras.layers import Input, LSTM, Dense, concatenate, Dropout
from keras.optimizers import Adam
from keras.models import Model
from tensorflow.keras.utils import to_categorical
from keras.callbacks.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# functions
def build_lstm(seq_shape, n_classes, hidden_units, lr):
	seq_layer = Input(shape=(seq_shape[0], seq_shape[1],))

	h = LSTM(hidden_units, return_sequences=True)(seq_layer)
	h = LSTM(hidden_units)(h)

	y = Dense(n_classes, activation='softmax')(h)

	adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999)
	model = Model(seq_layer, y)
	model.compile(loss='categorical_crossentropy',
		optimizer=adam,
		metrics=['acc'])
	return model

# params
lr = 0.001
n_classes = 3
hidden_units = 128
batch_size = 128
epochs = 400

if len(sys.argv) != 6:
	print('Usg: python {} train_x_series.npy train_y.csv model.json weights.h5 result.png'.format(sys.argv[0]))
	exit()
train_x_series_path = sys.argv[1]
train_y_path = sys.argv[2]
model_json_path = sys.argv[3]
model_weights_path = sys.argv[4]
result_image_path = sys.argv[5]

# load data
train_x_series = np.load(train_x_series_path)
valid_x_series = np.load('./data/valid_series.npy')

train_y = to_categorical(pd.read_csv(train_y_path)['roas'].values)
valid_y = to_categorical(pd.read_csv('./data/validset.csv')['roas'].values)

# build model and train
seq_shape = train_x_series.shape[1:]

model = build_lstm(seq_shape, n_classes, hidden_units, lr)

model_json = model.to_json()
with open(model_json_path, 'w') as json_file:
	json_file.write(model_json)

ckpt = ModelCheckpoint(filepath=model_weights_path, verbose=1, save_best_only=True)

history = model.fit(train_x_series,
	train_y,
	batch_size=batch_size,
	epochs=epochs,
	validation_data=(valid_x_series, valid_y),
	callbacks=[ckpt])

# plot learning curve
f = plt.figure()
ax = f.add_subplot(111)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(result_image_path)

