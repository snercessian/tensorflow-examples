#!/usr/bin/env python3

"""
Batch normalization example using MNIST
Illustrates improved accuracy/convergence of a simple network when batch normalization is applied

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
NUM_CLASSES     = 10

# architecture parameters
NUM_HIDDEN      = 100

# learning parameters
BATCH_SIZE      = 50
NUM_STEPS       = 10000
NUM_STEPS_OUT   = 100 # display accuracy every NUM_STEPS_OUT steps
LEARNING_RATE   = 1e-2
WEIGHT_DECAY    = 0   # regularization
EPSILON         = 1e-3

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
    y_      = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

    activation = tf.nn.relu
    
    ### model
    # w/o batch normalization
    W_fc1 = init_variable([NUM_PIXELS, NUM_HIDDEN])
    b_fc1 = init_variable([NUM_HIDDEN])
    z_fc1 = tf.matmul(x    , W_fc1) + b_fc1
    h_fc1 = activation(z_fc1)
    
    W_fc2 = init_variable([NUM_HIDDEN, NUM_HIDDEN])
    b_fc2 = init_variable([NUM_HIDDEN])
    z_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
    h_fc2 = activation(z_fc2)
                     
    W_fc3 = init_variable([NUM_HIDDEN, NUM_CLASSES])
    b_fc3 = init_variable([NUM_CLASSES])
    h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3
                     
    y_fc = h_fc3
    
    # w/ batch normalization
    W_fc1_bn    = init_variable([NUM_PIXELS, NUM_HIDDEN])
    b_fc1_bn    = init_variable([NUM_HIDDEN])
    z_fc1_bn    = tf.matmul(x    , W_fc1_bn) + b_fc1_bn
    mu_1, sig_1 = tf.nn.moments(z_fc1_bn,[0])
    scale_1     = tf.Variable(tf.ones( [NUM_HIDDEN]))
    beta_1      = tf.Variable(tf.zeros([NUM_HIDDEN]))
    z_fc1_bn_n  = tf.nn.batch_normalization(z_fc1_bn,mu_1,sig_1,beta_1,scale_1,EPSILON)
    h_fc1_bn    = activation(z_fc1_bn_n)
    
    W_fc2_bn    = init_variable([NUM_HIDDEN, NUM_HIDDEN])
    b_fc2_bn    = init_variable([NUM_HIDDEN])
    z_fc2_bn    = tf.matmul(h_fc1_bn, W_fc2_bn) + b_fc2_bn
    mu_2, sig_2 = tf.nn.moments(z_fc2_bn,[0])
    scale_2     = tf.Variable(tf.ones( [NUM_HIDDEN]))
    beta_2      = tf.Variable(tf.zeros([NUM_HIDDEN]))
    z_fc2_bn_n  = tf.nn.batch_normalization(z_fc2_bn,mu_2,sig_2,beta_2,scale_2,EPSILON)
    h_fc2_bn    = activation(z_fc2_bn_n)
                     
    W_fc3_bn = init_variable([NUM_HIDDEN, NUM_CLASSES])
    b_fc3_bn = init_variable([NUM_CLASSES])
    h_fc3_bn = tf.matmul(h_fc2_bn, W_fc3_bn) + b_fc3_bn
    
    y_fc_bn = h_fc3_bn
    
    ### cost function
    # w/o batch normalization
    cross_entropy  = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_fc))
    regularization = WEIGHT_DECAY*(tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc2)+tf.nn.l2_loss(W_fc3))
    loss           = cross_entropy+regularization
    correct_prediction = tf.equal(tf.argmax(y_fc,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    
        
    # w/ batch normalization
    cross_entropy_bn  = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_fc_bn))
    regularization_bn = WEIGHT_DECAY*(tf.nn.l2_loss(W_fc1_bn)+tf.nn.l2_loss(W_fc2_bn)+tf.nn.l2_loss(W_fc3_bn))
    loss_bn           = cross_entropy_bn+regularization_bn
    correct_prediction_bn = tf.equal(tf.argmax(y_fc_bn,1), tf.argmax(y_,1))
    accuracy_bn = tf.reduce_mean(tf.cast(correct_prediction_bn, tf.float32))    

    # optimization setup    
    train_step    = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    train_step_bn = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_bn)
    
    print('Model defined')
    
    # begin training
    print('Begin training model')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    acc = [] 
    acc_bn = []
    for i in range(NUM_STEPS):
        batch = mnist.train.next_batch(BATCH_SIZE)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        train_step_bn.run(feed_dict={x: batch[0], y_: batch[1]})
        if i%NUM_STEPS_OUT == 0:         
            accuracy_test     = accuracy.eval(feed_dict={x: mnist.test.images , y_: mnist.test.labels})
            accuracy_test_bn  = accuracy_bn.eval(feed_dict={x: mnist.test.images , y_: mnist.test.labels})
            acc.append(accuracy_test)
            acc_bn.append(accuracy_test_bn)
            plt.plot(acc,'b'),plt.plot(acc_bn,'r'), plt.show()
            print("Test accuracy w/o BN, w/ BN (step %d): %g, %g" % (i, accuracy_test, accuracy_test_bn))
        
main()