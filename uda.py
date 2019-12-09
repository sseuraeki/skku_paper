import sys
import tsaug
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import kullback_leibler_divergence as KL
from keras.losses import categorical_crossentropy
from keras.layers import Input, Conv1D, LeakyReLU, Flatten, Dropout, Dense
from keras.layers import LSTM, BatchNormalization, Activation, add, GlobalAveragePooling1D
import matplotlib.pyplot as plt

# parameters
batch_size = 128
hidden_units = 128
n_epochs = 10
learning_rate = 0.001

if len(sys.argv) != 8:
	print(
		'Usg: python {} train_x_series.npy train_y.csv model.json weights.h5 result.png cnn/lstm/resnet scale/jitter/time_warp/crop'.format(
			sys.argv[0]))
	exit()

if sys.argv[6] not in ['cnn', 'lstm', 'resnet']:
	print("ERROR: model type must be either one of ['cnn', 'lstm', 'resnet']")
	exit()

if sys.argv[7] not in ['scale', 'jitter', 'time_warp', 'crop']:
	print("ERROR: augment type must be either one of ['scale', 'jitter', 'time_warp', 'crop']")
	exit()

train_x_path = sys.argv[1]
train_y_path = sys.argv[2]
valid_x_path = 'data/valid_series.npy'
valid_y_path = 'data/validset.csv'
unlabeled_path = 'data/unlabeled.npy'

model_json_path = sys.argv[3]
model_weights_path = sys.argv[4]
result_image_path = sys.argv[5]
model_type = sys.argv[6]
augment_type = sys.argv[7]

# functions
def build_supervised_model(seq_shape, n_classes):
	# input layer
	labeled_input = Input(shape=(seq_shape[1], seq_shape[2],))

	# build architecture
	if model_type == 'cnn':
		# downsample (None, 4, 128)
		h = Conv1D(filters=hidden_units, kernel_size=3, strides=1, padding='same')(labeled_input)
		h = LeakyReLU(alpha=0.2)(h)

		# downsample (None, 4, 128)
		h = Conv1D(filters=hidden_units, kernel_size=3, strides=1, padding='same')(h)
		h = LeakyReLU(alpha=0.2)(h)

		# downsample (None, 4, 128)
		h = Conv1D(filters=hidden_units, kernel_size=3, strides=1, padding='same')(h)
		h = LeakyReLU(alpha=0.2)(h)

		# fully connect (None, 512)
		h = Flatten()(h)
		h = Dropout(0.4)(h)

	if model_type == 'lstm':
		h = LSTM(hidden_units, return_sequences=True)(labeled_input)
		h = LSTM(hidden_units)(h)

	if model_type == 'resnet':
		b1 = Conv1D(filters=hidden_units, kernel_size=8, padding='same')(labeled_input)
		b1 = BatchNormalization()(b1)
		b1 = Activation('relu')(b1)

		b1 = Conv1D(filters=hidden_units, kernel_size=5, padding='same')(b1)
		b1 = BatchNormalization()(b1)
		b1 = Activation('relu')(b1)

		b1 = Conv1D(filters=hidden_units, kernel_size=3, padding='same')(b1)
		b1 = BatchNormalization()(b1)

		shortcut = Conv1D(filters=hidden_units, kernel_size=1, padding='same')(labeled_input)
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
		h = GlobalAveragePooling1D()(b3)

	# compile supervised model
	output = Dense(n_classes, activation='softmax')(h)
	supervised_model = Model(labeled_input, output)
	supervised_model.compile(loss='categorical_crossentropy',
		optimizer=Adam(lr=learning_rate),
		metrics=['accuracy'])

	return supervised_model

def build_uda(supervised_model):
	uda = Model(supervised_model.input, supervised_model.output)
	uda.compile(loss='kullback_leibler_divergence',
		optimizer=Adam(lr=learning_rate),
		metrics=['accuracy'])
	return uda

def sample_batch(x, y, n_samples, n_batch):
	start_idx = n_batch * n_samples
	end_idx = (n_batch + 1) * n_samples
	if end_idx > x.shape[0]:
		end_idx = x.shape[0]

	x = x[start_idx:end_idx]
	if y is not None:
		y = y[start_idx:end_idx]
		return x, y
	return x

def sample_unsupervised(dataset, n_samples, n_batch):
	start_idx = n_batch * n_samples
	end_idx = (n_batch + 1) * n_samples
	if end_idx > dataset.shape[0]:
		end_idx = dataset.shape[0]

	x = dataset[start_idx:end_idx]
	return x

def augment(x, augment_type):
	if augment_type == 'scale':
		return tsaug.random_affine(x, max_a=1.1, min_a=0.1, max_b=0.01, min_b=-0.01)
	if augment_type == 'jitter':
		return tsaug.random_jitter(x)
	if augment_type == 'time_warp':
		return tsaug.random_time_warp(x)
	if augment_type == 'crop':
		x = tsaug.random_crop(x, crop_size=2)
		nan = np.zeros((x.shape))
		return np.concatenate((x, nan), axis=1)
	return None

def train(supervised_model, uda,
	train_x, train_y, valid_x, valid_y, unlabeled_x,
	n_epochs, supervised_batch_size):
	# calculate number of batches
	n_batches = int(np.ceil(train_x.shape[0] / supervised_batch_size))

	# calculate unsupervised batch size
	unsupervised_batch_size = unlabeled_x.shape[0] // n_batches

	# enumerate epochs
	all_train_accs = []
	all_valid_accs = []
	best_valid_loss = np.inf
	for n_epoch in range(n_epochs):
		train_losses = []
		train_accs = []
		for n_batch in range(n_batches):
			# update supervised model
			sup_x_batch, sup_y_batch = sample_batch(train_x, train_y, supervised_batch_size, n_batch)
			sup_loss, sup_acc = supervised_model.train_on_batch(sup_x_batch, sup_y_batch)
			train_losses.append(sup_loss)
			train_accs.append(sup_acc)

			# update uda
			unlabel_x_batch = sample_batch(unlabeled_x, None, unsupervised_batch_size, n_batch)
			aug_x_batch = augment(unlabel_x_batch, augment_type)
			uda.train_on_batch(aug_x_batch, supervised_model.predict(unlabel_x_batch))

		# test on validset
		score = supervised_model.evaluate(valid_x, valid_y)

		print('Epoch', n_epoch+1)
		print('Supervised loss:', np.mean(train_losses))
		print('Supervised acc:', np.mean(train_accs))
		print('Valid acc:', score[1])
		print('')

		all_train_accs.append(np.mean(train_accs))
		all_valid_accs.append(score[1])

		valid_loss = score[0]
		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			supervised_model.save_weights(model_weights_path)

	return all_train_accs, all_valid_accs

# load data
train_x = np.load(train_x_path)
valid_x = np.load(valid_x_path)
unlabeled_x = np.load(unlabeled_path)

train_y = to_categorical(pd.read_csv(train_y_path)['roas'].values)
valid_y = to_categorical(pd.read_csv(valid_y_path)['roas'].values)

# build model
n_classes = len(pd.read_csv(train_y_path)['roas'].unique())
supervised_model = build_supervised_model(train_x.shape, n_classes)
uda = build_uda(supervised_model)

model_json = supervised_model.to_json()
with open(model_json_path, 'w') as json_file:
	json_file.write(model_json)

# train
results = train(supervised_model, uda,
	train_x, train_y, valid_x, valid_y, unlabeled_x,
	n_epochs, batch_size)

# plot results
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
result1, = ax.plot(results[0], label='Trainset acc')
result2, = ax.plot(results[1], label='Validset acc')
ax.legend(loc='upper left')
plt.title('UDA results')
plt.savefig(result_image_path)





















