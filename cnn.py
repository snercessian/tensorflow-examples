#!/usr/bin/env python3

"""
Convolutional neural network example using MNIST


@author: shahannercessian
"""

#import
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# input/output sizes
IMAGE_SIZE      = 28 
NUM_PIXELS      = IMAGE_SIZE**2
NUM_CLASSES     = 10

# CNN architecture parameters
WS              = 5 
NUM_FILT1       = 32
NUM_FILT2       = 64
NUM_HIDDEN      = 1024

# learning parameters
BATCH_SIZE      = 50
NUM_STEPS       = 20000
NUM_STEPS_OUT   = 100 # display accuracy every NUM_STEPS_OUT steps
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4 # regularization

def init_variable(shape):
    return tf.Variable(tf.random_normal(shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main():
    # load MNIST dataset
    print('Loading data...')
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print('Data loaded')
    
    print('Defining model architecture')
    # input/output placeholders
    x       = tf.placeholder(tf.float32, shape=[None, NUM_PIXELS])
    x_image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,1])
    y_      = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

    # define activation
    activation = tf.nn.relu

    # convolutional + pooling layer 1   
    W_conv1 = init_variable([WS, WS, 1, NUM_FILT1])
    b_conv1 = init_variable([NUM_FILT1])
    
    h_conv1 = activation(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # convolutional + pooling layer 2  
    W_conv2 = init_variable([WS, WS, NUM_FILT1, NUM_FILT2])
    b_conv2 = init_variable([NUM_FILT2])
    
    h_conv2 = activation(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    # densely connected layer
    dims = h_pool2.get_shape().as_list() ; 
                            
    NUM_FEATS = dims[1]*dims[2]*dims[3] ;
    print(NUM_FEATS)
    h_pool2_flat = tf.reshape(h_pool2, [-1, NUM_FEATS]) # vectorize feature maps
    
    W_fc1 = init_variable([NUM_FEATS, NUM_HIDDEN])
    b_fc1 = init_variable([NUM_HIDDEN])
    
    h_fc1 = activation(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # dropout on densely connected layer
    keep_prob  = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = init_variable([NUM_HIDDEN, NUM_CLASSES])
    b_fc2 = init_variable([NUM_CLASSES])
    
    # output layer
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
                      
    # cost function
    cross_entropy  = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    regularization = WEIGHT_DECAY*(tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc2))
    loss           = cross_entropy+regularization
    
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # optimization setup
    train_step    = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    
    print('Model defined')
    
    # begin training
    print('Begin training model')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    for i in range(NUM_STEPS):
        batch = mnist.train.next_batch(BATCH_SIZE)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i%NUM_STEPS_OUT == 0:         
            accuracy_batch = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("Batch accuracy (step %d): %g" % (i, accuracy_batch))
            
    accuracy_train = accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1.0})
    accuracy_test  = accuracy.eval(feed_dict={x: mnist.test.images , y_: mnist.test.labels , keep_prob: 1.0})
    print("Train accuracy (step %d): %g" % (i, accuracy_train))
    print("Test  accuracy (step %d): %g" % (i, accuracy_test ))
        
main()