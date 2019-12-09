import sys
import numpy as np
import pandas as pd
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Conv1D, Activation
from keras.layers import add, GlobalAveragePooling1D, Dense
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# parameters
hidden_units = 128
n_classes = 3
learning_rate = 0.001
min_lr = 0.0001
batch_size = 64
epochs = 1000

if len(sys.argv) != 6:
	print('Usg: python {} train_x_series.npy train_y.csv model.json weights.h5 result.png'.format(sys.argv[0]))
	exit()
train_x_series_path = sys.argv[1]
train_y_path = sys.argv[2]
model_json_path = sys.argv[3]
model_weights_path = sys.argv[4]
result_image_path = sys.argv[5]

# functions
def build_model(seq_shape, n_classes):
	# Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline, Wang et al, 2016
	# https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py
	seq_layer = Input(shape=(seq_shape[1], seq_shape[2],))

	# block 1
	b1 = Conv1D(filters=hidden_units, kernel_size=8, padding='same')(seq_layer)
	b1 = BatchNormalization()(b1)
	b1 = Activation('relu')(b1)

	b1 = Conv1D(filters=hidden_units, kernel_size=5, padding='same')(b1)
	b1 = BatchNormalization()(b1)
	b1 = Activation('relu')(b1)

	b1 = Conv1D(filters=hidden_units, kernel_size=3, padding='same')(b1)
	b1 = BatchNormalization()(b1)

	shortcut = Conv1D(filters=hidden_units, kernel_size=1, padding='same')(seq_layer)
	shortcut = BatchNormalization()(shortcut)

	b1 = add([shortcut, b1])
	b1 = Activation('relu')(b1)

	# block 2
	b2 = Conv1D(filters=hidden_units*2, kernel_size=8, padding='same')(b1)
	b2 = BatchNormalization()(b2)
	b2 = Activation('relu')(b2)

	b2 = Conv1D(filters=hidden_units*2, kernel_size=5, padding='same')(b2)
	b2 = BatchNormalization()(b2)
	b2 = Activation('relu')(b2)

	b2 = Conv1D(filters=hidden_units*2, kernel_size=3, padding='same')(b2)
	b2 = BatchNormalization()(b2)

	shortcut = Conv1D(filters=hidden_units*2, kernel_size=1, padding='same')(b1)
	shortcut = BatchNormalization()(shortcut)

	b2 = add([shortcut, b2])
	b2 = Activation('relu')(b2)

	# block 3
	b3 = Conv1D(filters=hidden_units*2, kernel_size=8, padding='same')(b2)
	b3 = BatchNormalization()(b3)
	b3 = Activation('relu')(b3)

	b3 = Conv1D(filters=hidden_units*2, kernel_size=5, padding='same')(b3)
	b3 = BatchNormalization()(b3)
	b3 = Activation('relu')(b3)

	b3 = Conv1D(filters=hidden_units*2, kernel_size=3, padding='same')(b3)
	b3 = BatchNormalization()(b3)

	shortcut = BatchNormalization()(b2)

	b3 = add([shortcut, b3])
	b3 = Activation('relu')(b3)

	# output
	pooling = GlobalAveragePooling1D()(b3)
	output_layer = Dense(n_classes, activation='softmax')(pooling)

	# compile
	model = Model(inputs=seq_layer, outputs=output_layer)
	model.compile(loss='categorical_crossentropy',
		optimizer=Adam(lr=learning_rate, clipnorm=1),
		metrics=['accuracy'])

	return model

# load data
train_x = np.load(sys.argv[1])
valid_x = np.load('./data/valid_series.npy')

train_y = to_categorical(pd.read_csv(sys.argv[2])['roas'].values)
valid_y = to_categorical(pd.read_csv('./data/validset.csv')['roas'].values)

# build model and train
seq_shape = train_x.shape

model = build_model(seq_shape, n_classes)

model_json = model.to_json()
with open(model_json_path, 'w') as json_file:
	json_file.write(model_json)

ckpt = ModelCheckpoint(filepath=model_weights_path, verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=min_lr)

history = model.fit(train_x,
	train_y,
	batch_size=batch_size,
	epochs=epochs,
	validation_data=(valid_x, valid_y),
	callbacks=[ckpt, reduce_lr])

# plot learning curve
f = plt.figure()
ax = f.add_subplot(111)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(result_image_path)





