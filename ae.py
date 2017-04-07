#!/usr/bin/env python3

"""
Auto-encoder with tied weights using MNIST


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

# auto-encoder architecture parameters
NUM_HIDDEN1     = 256
NUM_HIDDEN2     = 64

# learning parameters
BATCH_SIZE      = 50
NUM_STEPS       = 10000
NUM_STEPS_OUT   = 100 # display accuracy every NUM_STEPS_OUT steps
LEARNING_RATE   = 1e-2
WEIGHT_DECAY    = 0   # regularization

def init_variable(shape):
    return tf.Variable(tf.random_normal(shape))

def main():
    # load MNIST dataset
    print('Loading data...')
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print('Data loaded')
    
    print('Defining model architecture')
    # input/output placeholders
    x       = tf.placeholder(tf.float32, shape=[None, NUM_PIXELS])
    y_      = tf.placeholder(tf.float32, shape=[None, NUM_PIXELS])

    # define activation
    activation = tf.nn.sigmoid
    
    # encoder
    W_fc1 = init_variable([NUM_PIXELS, NUM_HIDDEN1])
    b_fc1 = init_variable([NUM_HIDDEN1])
    h_fc1 = activation(tf.matmul(x    , W_fc1) + b_fc1)

    W_fc2 = init_variable([NUM_HIDDEN1, NUM_HIDDEN2])
    b_fc2 = init_variable([NUM_HIDDEN2])
    h_fc2 = activation(tf.matmul(h_fc1, W_fc2) + b_fc2)    
    
    # decoder
    W_fc3 = tf.transpose(W_fc2)
    b_fc3 = init_variable([NUM_HIDDEN1])
    h_fc3 = activation(tf.matmul(h_fc2, W_fc3) + b_fc3)   
    
    W_fc4 = tf.transpose(W_fc1)
    b_fc4 = init_variable([NUM_PIXELS])
    h_fc4 = activation(tf.matmul(h_fc3, W_fc4) + b_fc4)   
    
    y_auto = h_fc4
    
    # cost function
    mse            = tf.reduce_sum(tf.square(y_ - y_auto))
    regularization = WEIGHT_DECAY*(tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc2))
    loss           = mse+regularization
    
    # optimization setup
    train_step    = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
            
    print('Model defined')
    
    # begin training
    print('Begin training model')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    ex1 = [mnist.train.images[0,:]]
    for i in range(NUM_STEPS):
        batch = mnist.train.next_batch(BATCH_SIZE)
        train_step.run(feed_dict={x: batch[0], y_: batch[0]})
        if i%NUM_STEPS_OUT == 0:         
            mse_batch = mse.eval(feed_dict={x: batch[0], y_: batch[0]})
            print("Batch reconstruction MSE (step %d): %g" % (i, mse_batch))
            
            ex2   = h_fc1.eval(feed_dict={x: ex1}) 
            ex3   = h_fc2.eval(feed_dict={x: ex1})
            ex4   = h_fc3.eval(feed_dict={x: ex1})
            ex5   = h_fc4.eval(feed_dict={x: ex1})
            plt.subplot(1,5,1)
            plt.imshow(np.reshape(ex1, (28,28)))
            plt.subplot(1,5,2)
            plt.imshow(np.reshape(ex2, (16,16)))
            plt.subplot(1,5,3)
            plt.imshow(np.reshape(ex3, (8,8)))
            plt.subplot(1,5,4)
            plt.imshow(np.reshape(ex4, (16,16)))
            plt.subplot(1,5,5)
            plt.imshow(np.reshape(ex5, (28,28)))
            plt.show()
            
    mse_train = mse.eval(feed_dict={x: mnist.train.images, y_: mnist.train.images})
    mse_test  = mse.eval(feed_dict={x: mnist.test.images , y_: mnist.test.images })
    print("Train reconstruction MSE (step %d): %g" % (i, mse_train))
    print("Test  reconstruction MSE (step %d): %g" % (i, mse_test ))
        
main()