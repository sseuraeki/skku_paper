import numpy as np
import pandas as pd
from keras.layers import Input, LSTM, Dense, concatenate, Dropout
from keras.optimizers import Adam
from keras.models import Model
from tensorflow.keras.utils import to_categorical
from keras.callbacks.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# functions
def build_lstm(seq_shape, plain_shape, n_classes, hidden_units, lr):
	seq_layer = Input(shape=(seq_shape[0], seq_shape[1],))
	plain_layer = Input(shape=(plain_shape,))

	h = LSTM(hidden_units)(seq_layer)
	#h = concatenate([h, plain_layer])
	#h = Dense(hidden_units // 2, activation='relu')(h)
	#h = Dropout(0.4)(h)

	y = Dense(n_classes, activation='softmax')(h)

	adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999)
	model = Model([seq_layer, plain_layer], y)
	model.compile(loss='categorical_crossentropy',
		optimizer=adam,
		metrics=['acc'])
	return model

# params
weights_filepath = './lstm_weights.hdf5'
lr = 0.001
n_classes = 3
hidden_units = 128

# load data
train_x_series = np.load('./data/train_series.npy')
valid_x_series = np.load('./data/valid_series.npy')
test_x_series = np.load('./data/test_series.npy')

train_x_plain = pd.read_csv('./data/trainset.csv')[['spend', 'install']].values
valid_x_plain = pd.read_csv('./data/validset.csv')[['spend', 'install']].values
test_x_plain = pd.read_csv('./data/testset.csv')[['spend', 'install']].values

train_y = to_categorical(pd.read_csv('./data/trainset.csv')['roas'].values)
valid_y = to_categorical(pd.read_csv('./data/validset.csv')['roas'].values)
test_y = to_categorical(pd.read_csv('./data/testset.csv')['roas'].values)

# build model and train
seq_shape = train_x_series.shape[1:]
plain_shape = train_x_plain.shape[1]

model = build_lstm(seq_shape, plain_shape, n_classes, hidden_units, lr)
ckpt = ModelCheckpoint(filepath=weights_filepath, verbose=1, save_best_only=True)

history = model.fit([train_x_series, train_x_plain],
	train_y,
	batch_size=128,
	epochs=500,
	validation_data=([valid_x_series, valid_x_plain], valid_y),
	callbacks=[ckpt])

# predict
model.load_weights(weights_filepath)
predictions = model.predict([test_x_series, test_x_plain])

# acc
correct = 0
for i in range(test_y.shape[0]):
	if np.argmax(predictions[i]) == np.argmax(test_y[i]):
		correct += 1

test_acc = (correct / len(test_y)) // 0.01 * 0.01

# plot learning curve
f = plt.figure()
ax = f.add_subplot(111)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.text(0.02, 0.84,
	'testset accuracy: {}'.format(test_acc),
	ha='left', va='top',
	transform=ax.transAxes)
plt.show()
