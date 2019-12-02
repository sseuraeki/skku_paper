import numpy as np
import pandas as pd
from keras import backend
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Input, Conv1D, LeakyReLU, Flatten
from keras.layers import Dropout, Dense, Activation, Lambda
from keras.layers import Reshape, Conv2DTranspose
import matplotlib.pyplot as plt

# maybe sample supervised balance?

# parameters
batch_size = 32
hidden_units = 128
n_epochs = 5
learning_rate = 0.0006
beta_1 = 0.5
latent_dim = 100
n_classes = 5
sup_train_x_path = './data/train_series.npy'
sup_train_y_path = './data/trainset.csv'
sup_valid_x_path = './data/valid_series.npy'
sup_valid_y_path = './data/validset.csv'
sup_test_x_path = './data/test_series.npy'
sup_test_y_path = './data/testset.csv'
unsup_path = './data/unlabeled.npy'
model_path = './sgan.h5'

# functions
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
	# https://stackoverflow.com/questions/44061208/how-to-implement-the-conv1dtranspose-in-keras
    x = Lambda(lambda x: backend.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: backend.squeeze(x, axis=2))(x)
    return x

def custom_activation(logits):
	# Improved Techniques for Training GANs, OpenAI
	logexpsum = backend.sum(backend.exp(logits), axis=-1, keepdims=True)
	result = logexpsum / (logexpsum + 1.0)
	return result

def build_discriminator(seq_shape, n_classes, hidden_units=128):
	# input (None, 4, 8)
	seq_layer = Input(shape=(seq_shape[1], seq_shape[2],))

	# downsample (None, 4, 128)
	h = Conv1D(filters=hidden_units, kernel_size=3, strides=1, padding='same')(seq_layer)
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
	h = Dense(n_classes)(h)

	# supervised output
	sup_layer = Activation('softmax')(h)

	# build supervised discriminator
	sup_model = Model(seq_layer, sup_layer)
	sup_model.compile(loss='categorical_crossentropy',
		optimizer=Adam(lr=learning_rate, beta_1=beta_1),
		metrics=['accuracy'])

	# unsupervised output
	unsup_layer = Lambda(custom_activation)(h)

	# build unsupervised discriminator
	unsup_model = Model(seq_layer, unsup_layer)
	unsup_model.compile(loss='binary_crossentropy',
		optimizer=Adam(lr=learning_rate, beta_1=beta_1),
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
	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate, beta_1=beta_1))
	return model

def sample_supervised(x, y, n_samples, n_classes):
	start_idx = n_batch * n_samples
	end_idx = (n_batch + 1) * n_samples
	if end_idx > x.shape[0]:
		end_idx = x.shape[0]

	x = x[start_idx:end_idx]
	y = y[start_idx:end_idx]
	return x, y

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

def train(generator, sup_model, unsup_model, gan,
	sup_x, sup_y, unsup_x, valid_x, valid_y,
	latent_dim, n_epochs, sup_batch_size):
	# calculate number of batches
	n_batches = int(np.ceil(sup_x.shape[0] / sup_batch_size))

	# calculate unsupervised batch size
	unsup_batch_size = unsup_x.shape[0] // n_batches

	# enumerate epochs
	all_sup_losses = []
	all_sup_accs = []
	all_unsup_losses = []
	all_gan_losses = []
	all_valid_accs = []
	best_valid_acc = 0.0
	for n_epoch in range(n_epochs):
		sup_losses = []
		sup_accs = []
		unsup_losses = []
		gan_losses = []		
		for n_batch in range(n_batches):
			# update supervised discriminator
			sup_x_batch, sup_y_batch = sample_supervised(sup_x, sup_y, sup_batch_size, n_batch)
			sup_loss, sup_acc = sup_model.train_on_batch(sup_x_batch, sup_y_batch)
			sup_losses.append(sup_loss)
			sup_accs.append(sup_acc)

			# update unsupervised discriminator
			x_real, y_real = generate_unsupervised_real(unsup_x, unsup_batch_size, n_batch)
			unsup_loss1 = unsup_model.train_on_batch(x_real, y_real)
			x_fake, y_fake = generate_unsupervised_fake(generator, latent_dim, unsup_batch_size)
			unsup_loss2 = unsup_model.train_on_batch(x_fake, y_fake)
			unsup_loss = unsup_loss1 + unsup_loss2
			unsup_losses.append(unsup_loss)

			# update generator
			x_gan, y_gan = generate_latent_points(latent_dim, unsup_batch_size), np.ones((unsup_batch_size, 1))
			gan_loss = gan.train_on_batch(x_gan, y_gan)
			gan_losses.append(gan_loss)

		# test on validset
		predictions = sup_model.predict(valid_x)
		correct = 0
		for i in range(predictions.shape[0]):
			if np.argmax(predictions[i]) == np.argmax(valid_y[i]):
				correct += 1
		valid_acc = correct / predictions.shape[0]

		print('Epoch', n_epoch+1)
		print('Supervised loss:', np.mean(sup_losses))
		print('Supervised acc:', np.mean(sup_accs))
		print('Unsupervised loss:', np.mean(unsup_losses))
		print('GAN loss:', np.mean(gan_losses))
		print('Valid acc:', valid_acc)
		print('')

		all_sup_losses.append(np.mean(sup_losses))
		all_sup_accs.append(np.mean(sup_accs))
		all_unsup_losses.append(np.mean(unsup_losses))
		all_gan_losses.append(np.mean(gan_losses))
		all_valid_accs.append(valid_acc)

		if valid_acc > best_valid_acc:
			best_valid_acc = valid_acc
			sup_model.save_weights(model_path)

	return all_sup_losses, all_sup_accs, all_unsup_losses, all_gan_losses, all_valid_accs

# load data
sup_train_x = np.load(sup_train_x_path)
sup_train_y = to_categorical(pd.read_csv(sup_train_y_path)['roas'].values)

sup_valid_x = np.load(sup_valid_x_path)
sup_valid_y = to_categorical(pd.read_csv(sup_valid_y_path)['roas'].values)

sup_test_x = np.load(sup_test_x_path)
sup_test_y = to_categorical(pd.read_csv(sup_test_y_path)['roas'].values)

unsup_x = np.random.shuffle(np.load(unsup_path))

# build models
seq_shape = sup_train_x.shape
sup_model, unsup_model = build_discriminator(seq_shape, n_classes, hidden_units=hidden_units)

generator = build_generator(latent_dim, seq_shape)

gan = build_gan(generator, unsup_model)

# train model
results = train(generator, sup_model, unsup_model, gan,
	sup_train_x, sup_train_y, unsup_x, sup_valid_x, sup_valid_y,
	latent_dim, n_epochs=n_epochs, sup_batch_size=batch_size)

# load weights and test
sup_model.load_weights(model_path)
predictions = sup_model.predict(sup_test_x)
correct = 0
for i in range(sup_test_y.shape[0]):
	if np.argmax(predictions[i]) == np.argmax(sup_test_y[i]):
		correct += 1
test_acc = correct / sup_test_y.shape[0]
print('Test accuracy:', test_acc)

# plot results
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
result1, = ax.plot(results[0], label='Supervised loss')
result2, = ax.plot(results[1], label='Trainset acc')
result3, = ax.plot(results[2], label='Unsupervised loss')
result4, = ax.plot(results[3], label='GAN loss')
result5, = ax.plot(results[4], label='Validset acc')
ax.legend(loc='upper left')
plt.title('SGAN results')
plt.show()




















