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
from keras.layers import Lambda, Conv2DTranspose, Reshape
import matplotlib.pyplot as plt

# parameters
batch_size = 128
hidden_units = 128
n_epochs = 10
learning_rate = 0.001
latent_dim = 100

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
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
	# https://stackoverflow.com/questions/44061208/how-to-implement-the-conv1dtranspose-in-keras
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def custom_activation(logits):
	# Improved Techniques for Training GANs, OpenAI
	logexpsum = K.sum(K.exp(logits), axis=-1, keepdims=True)
	result = logexpsum / (logexpsum + 1.0)
	return result

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

	h = Dense(n_classes)(h)

	# supervised output
	sup_layer = Activation('softmax')(h)

	# build supervised discriminator
	sup_model = Model(labeled_input, sup_layer)
	sup_model.compile(loss='categorical_crossentropy',
		optimizer=Adam(lr=learning_rate),
		metrics=['accuracy'])

	# unsupervised output
	unsup_layer = Lambda(custom_activation)(h)

	# build unsupervised discriminator
	unsup_model = Model(labeled_input, unsup_layer)
	unsup_model.compile(loss='binary_crossentropy',
		optimizer=Adam(lr=learning_rate),
		metrics=['accuracy'])

	return sup_model, unsup_model

def build_generator(latent_dim, seq_shape):
	# input (None, 100)
	input_layer = Input(shape=(latent_dim,))

	# starting layer (None, 256) > (None, 8, 32)
	nodes = seq_shape[1] * 2 * seq_shape[2] * 4
	h = Dense(nodes, activation='relu')(input_layer)
	h = LeakyReLU(alpha=0.2)(h)
	h = Reshape((seq_shape[1] * 2, seq_shape[2] * 4))(h)

	# upsample (None, 16, 128)
	h = Conv1DTranspose(h, filters=128, kernel_size=4, strides=2, padding='same')
	h = LeakyReLU(alpha=0.2)(h)

	# upsample (None, 32, 128)
	h = Conv1DTranspose(h, filters=128, kernel_size=4, strides=2, padding='same')
	h = LeakyReLU(alpha=0.2)(h)

	# output (None, 4, 8)
	out_layer = Conv1D(filters=8, kernel_size=8, strides=8, padding='same', activation='tanh')(h)

	# build generator
	model = Model(input_layer, out_layer)
	return model

def build_gan(generator, unsup_model):
	# hold discriminator weights
	unsup_model.trainable = False

	# generator's output > discriminator's input
	gan_output = unsup_model(generator.output)

	# build gan
	model = Model(generator.input, gan_output)
	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate))
	return model

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

def generate_unsupervised_real(dataset, n_samples, n_batch):
	start_idx = n_batch * n_samples
	end_idx = (n_batch + 1) * n_samples
	if end_idx > dataset.shape[0]:
		end_idx = dataset.shape[0]

	x = dataset[start_idx:end_idx]
	y = np.ones((len(x),1))
	return x, y

def generate_latent_points(latent_dim, n_samples):
	z_input = np.random.randn(latent_dim * n_samples)
	z_input = z_input.reshape(n_samples, latent_dim)
	return z_input

def generate_unsupervised_fake(generator, latent_dim, n_samples):
	z_input = generate_latent_points(latent_dim, n_samples)
	x = generator.predict(z_input)
	y = np.zeros((n_samples,1))
	return x, y

def train(supervised_model, uda, unsupervised_model, gan,
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

			# update unsupervised discriminator
			x_real, y_real = generate_unsupervised_real(unlabeled_x, unsupervised_batch_size, n_batch)
			unsup_loss1 = unsupervised_model.train_on_batch(x_real, y_real)
			x_fake, y_fake = generate_unsupervised_fake(generator, latent_dim, unsupervised_batch_size)
			unsup_loss2 = unsupervised_model.train_on_batch(x_fake, y_fake)

			# update generator
			x_gan = generate_latent_points(latent_dim, unsupervised_batch_size)
			y_gan = np.ones((unsupervised_batch_size, 1))
			gan_loss = gan.train_on_batch(x_gan, y_gan)

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
supervised_model, unsupervised_model = build_supervised_model(train_x.shape, n_classes)
uda = build_uda(supervised_model)
generator = build_generator(latent_dim, train_x.shape)
gan = build_gan(generator, unsupervised_model)

model_json = supervised_model.to_json()
with open(model_json_path, 'w') as json_file:
	json_file.write(model_json)

# train
results = train(supervised_model, uda, unsupervised_model, gan,
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





















