#!/usr/bin/env python3

"""
Convolutional auto-encoder with tied weights on MNIST
Encoder reflects convolutional stages of cnn.py

@author: shahannercessian
"""

#import utils
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# input/output sizes
IMAGE_SIZE      = 28 
NUM_PIXELS      = IMAGE_SIZE**2

# CAE architecture parameters
WS              = 5 
NUM_FILT1       = 32
NUM_FILT2       = 64

# learning parameters
BATCH_SIZE      = 50
NUM_STEPS       = 20000
NUM_STEPS_OUT   = 100 # display accuracy every NUM_STEPS_OUT steps
LEARNING_RATE   = 1e-2
SPARSITY        = 1e-4 # regularization

def init_variable(shape):
    return tf.Variable(tf.random_normal(shape))

def conv2d_2x2(x, W): # use strided convolutions instead of max-pooling
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def conv2d_transpose_2x2(x, W, shape): 
    return tf.nn.conv2d_transpose(x, W, shape, strides=[1, 2, 2, 1], padding='SAME')

def main():
    # load MNIST dataset
    print('Loading data...')
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print('Data loaded')
    
    print('Defining model architecture')
    # input/output placeholders
    x       = tf.placeholder(tf.float32, shape=[None, NUM_PIXELS])
    x_image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,1])
    y_      = tf.placeholder(tf.float32, shape=[None, NUM_PIXELS])

    SAMPLE_SIZE = tf.shape(x_image)[0]
    
    # define activation
    activation = tf.nn.sigmoid

    # Encoder
    # convolutional + upsampling layer 1   
    W_conv1 = init_variable([WS, WS, 1, NUM_FILT1])
    b_conv1 = init_variable([NUM_FILT1])
    h_pool1 = activation(conv2d_2x2(x_image, W_conv1) + b_conv1)

    # convolutional + upsampling layer 2  
    W_conv2 = init_variable([WS, WS, NUM_FILT1, NUM_FILT2])
    b_conv2 = init_variable([NUM_FILT2])
    h_pool2 = activation(conv2d_2x2(h_pool1, W_conv2) + b_conv2)

    # Decoder
    SHAPE = h_pool1.get_shape().as_list() ; SHAPE[0]=SAMPLE_SIZE
    b_conv3 = init_variable([NUM_FILT1])
    h_pool3 = activation(conv2d_transpose_2x2(h_pool2,W_conv2,tf.stack(SHAPE)) + b_conv3)
    
    SHAPE = x_image.get_shape().as_list() ; SHAPE[0]=SAMPLE_SIZE
    b_conv4 = init_variable([1])
    h_pool4 = activation(conv2d_transpose_2x2(h_pool3,W_conv1,tf.stack(SHAPE)) + b_conv4)
    
    y_auto  = tf.reshape(h_pool4, [-1,NUM_PIXELS])
    
    # cost function
    mse            = tf.reduce_sum(tf.square(y_ - y_auto))
    regularization = SPARSITY*tf.reduce_sum(tf.abs(h_pool2)) # could have also used KL divergence
    loss           = mse+regularization
    
    # optimization setup
    train_step    = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        
    print('Model defined')
    
    # begin training
    print('Begin training model')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    ex = [mnist.train.images[0,:]]
    for i in range(NUM_STEPS):
        batch = mnist.train.next_batch(BATCH_SIZE)
        train_step.run(feed_dict={x: batch[0], y_: batch[0]})
        if i%NUM_STEPS_OUT == 0:         
            mse_batch = mse.eval(feed_dict={x: batch[0], y_: batch[0]})
            print("Batch reconstruction MSE (step %d): %g" % (i, mse_batch))
            
            ex_recon = h_pool4.eval(feed_dict={x: ex})

            plt.subplot(1,2,1)
            plt.imshow(np.reshape(ex, (28,28)))
            plt.subplot(1,2,2)
            plt.imshow(np.reshape(ex_recon, (28,28)))
            plt.show()
            
    mse_train = mse.eval(feed_dict={x: mnist.train.images, y_: mnist.train.images})
    mse_test  = mse.eval(feed_dict={x: mnist.test.images , y_: mnist.test.images })
    print("Train reconstruction MSE (step %d): %g" % (i, mse_train))
    print("Test  reconstruction MSE (step %d): %g" % (i, mse_test ))
    
main()