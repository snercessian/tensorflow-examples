#!/usr/bin/env python3

"""
Recurrent neural network (LSTM) classification example using MNIST


@author: shahannercessian
"""

#import
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# input/output sizes
IMAGE_SIZE      = 28 
NUM_PIXELS      = IMAGE_SIZE**2
NUM_CLASSES     = 10

# LSTM architecture parameters
NUM_INPUTS    = IMAGE_SIZE
NUM_TIMESTEPS = IMAGE_SIZE
NUM_HIDDEN    = 128

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
    x       = tf.placeholder(tf.float32, shape=[None, NUM_TIMESTEPS, NUM_INPUTS])
    y_      = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
    
    # Fully connected layers
    W_fc1 = init_variable([NUM_INPUTS, NUM_HIDDEN])
    b_fc1 = init_variable([NUM_HIDDEN])
    W_fc2 = init_variable([NUM_HIDDEN, NUM_CLASSES])
    b_fc2 = init_variable([NUM_CLASSES])
    
    def RNN(x,W_fc1,b_fc1,W_fc2,b_fc2):
        # Fully connected layer
        x = tf.reshape(x,[-1,NUM_INPUTS])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1 = tf.reshape(h_fc1, [-1, NUM_TIMESTEPS, NUM_HIDDEN])
        
        # LSTM cell
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN, forget_bias=1.0, state_is_tuple=True)
        init_state = lstm_cell.zero_state(BATCH_SIZE,dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell,h_fc1, initial_state=init_state, time_major=False)
        
        # Output layer
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
        h_fc2 = tf.matmul(outputs[-1],W_fc2)+b_fc2

        return h_fc2
        
    y_lstm = RNN(x,W_fc1,b_fc1,W_fc2,b_fc2)
    # cost function
    cross_entropy  = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_lstm))
    regularization = WEIGHT_DECAY*(tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc2))
    loss           = cross_entropy+regularization
    
    correct_prediction = tf.equal(tf.argmax(y_lstm,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # optimization setup
    train_step    = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    
    print('Model defined')
    # begin training
    print('Begin training model')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    for i in range(NUM_STEPS):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        batch_x = batch_x.reshape([BATCH_SIZE, NUM_TIMESTEPS, NUM_INPUTS]) # turn batch into a data sequence
        train_step.run(feed_dict={x: batch_x, y_: batch_y})
        if i%NUM_STEPS_OUT == 0:         
            accuracy_batch = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
            print("Batch accuracy (step %d): %g" % (i, accuracy_batch))
    
    test_x  = mnist.test.images[:BATCH_SIZE].reshape( (-1, NUM_TIMESTEPS, NUM_INPUTS))
    test_y  = mnist.test.labels[:BATCH_SIZE]
    accuracy_test  = accuracy.eval(feed_dict={x:  test_x, y_:  test_y})
    print("Test  accuracy (step %d): %g" % (i, accuracy_test ))

main()