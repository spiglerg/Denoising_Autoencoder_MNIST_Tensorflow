import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 64
display_step = 1
examples_to_show = 10

is_denoising = True


# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)


# Build a (Denoising) Autoencoder with given hidden layers dimensions
def build_autoencoder(input_tensor, dimensions=[512, 256, 64]):
	corrupt_input = tf.placeholder_with_default(tf.constant(False), []) # False=don't corrupt, True=corrupt
	noise = tf.cast(tf.random_uniform(shape=tf.shape(input_tensor),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32)

	# Corrupt the input for the training phase of the Denoising Autoencoder
	last_input = tf.cond(corrupt_input,   lambda: input_tensor*noise,   lambda: input_tensor)

	weights = []
	# Build the encoder
	for size in dimensions:
		input_size = int(last_input.get_shape()[1])
		output_size = size

		w = tf.Variable( tf.random_uniform([input_size,output_size], -1.0/np.sqrt(input_size), 1.0/np.sqrt(input_size)) )
		weights.append(w)

		b = tf.Variable( tf.zeros([output_size]) )

		enc_layer = tf.nn.relu( tf.matmul(last_input, w) + b )
		last_input = enc_layer

	# Build the decoder
	dimensions.insert( 0, int(input_tensor.get_shape()[1]) )
	for i in range(len(dimensions)-2, -1, -1):
		input_size = int(last_input.get_shape()[1])
		output_size = dimensions[i]

		w = tf.transpose(weights[i])
		b = tf.Variable( tf.zeros([output_size]) )

		dec_layer = tf.nn.relu( tf.matmul(last_input, w) + b )
		last_input = dec_layer

	return last_input, corrupt_input




x = tf.placeholder("float", [None, n_input])
x_reconstructed, corrupt_input = build_autoencoder(x, [512, 256, 64])


# Define loss and optimizer (minimize the reconstruction error)
cost = tf.reduce_mean(tf.square(x_reconstructed-x))
optimize = tf.train.AdamOptimizer(learning_rate).minimize(cost)



# Initialize a graph
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


# Training loop
for epoch in range(training_epochs):
	# Loop over all batches
		for i in range( int(mnist.train.num_examples/batch_size) ):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)

			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([optimize, cost], feed_dict={x: batch_xs, corrupt_input:is_denoising}) # corrupt_input is only used in denoising autoencoders

		# Display logs per epoch step
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1),
				"cost=", "{:.9f}".format(c))


# Display some reconstructed inputs from the test set
xs = mnist.test.images[:examples_to_show]
encode_decode = sess.run(
		x_reconstructed, feed_dict={x: xs, corrupt_input:False})

# Compare original images with their reconstructions
f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
for i in range(examples_to_show):
		a[0][5].set_title('Original')
		a[1][5].set_title('Reconstructed')


		a[0][i].imshow(np.reshape(xs[i], (28, 28)))
		a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
		a[0][i].axis('off')
		a[1][i].axis('off')
f.show()
plt.draw()
plt.waitforbuttonpress()



