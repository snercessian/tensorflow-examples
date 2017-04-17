#!/usr/bin/env python3

"""
Recurrent neural network (LSTM) prediction example using MNIST


@author: shahannercessian
"""

#import
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# input/output sizes
IMAGE_SIZE      = 28 
NUM_PIXELS      = IMAGE_SIZE**2
NUM_OUTPUTS     = IMAGE_SIZE

# LSTM architecture parameters
NUM_INPUTS    = IMAGE_SIZE
NUM_TIMESTEPS = 14 # use half of the image to predict the second half
NUM_HIDDEN    = 128

# learning parameters
BATCH_SIZE      = 50
NUM_STEPS       = 20000
NUM_STEPS_OUT   = 100 # display accuracy every NUM_STEPS_OUT steps
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4 # regularization

def init_variable(shape):
    return tf.Variable(tf.random_normal(shape))

def gen_training_data(images,NUM_TIMESTEPS):
    
    x = []
    y = []
    
    for im in images:
        im = im.reshape((IMAGE_SIZE,IMAGE_SIZE))
        for k in range(IMAGE_SIZE-NUM_TIMESTEPS):
            x_seq = im[k:k+NUM_TIMESTEPS,:] # NUM_TIMESTEPS row sequence
            y_seq = im[k+NUM_TIMESTEPS,:]   # single row output
            x.append(x_seq),y.append(y_seq)
    
    batch_x = np.asarray(x)
    batch_y = np.asarray(y)
    return batch_x, batch_y
    
def main():
    # load MNIST dataset
    print('Loading data...')
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print('Data loaded')
    
    print('Defining model architecture')
    # input/output placeholders
    x       = tf.placeholder(tf.float32, shape=[None, NUM_TIMESTEPS, NUM_INPUTS])
    y_      = tf.placeholder(tf.float32, shape=[None, NUM_OUTPUTS])
    
    SAMPLE_SIZE = tf.shape(x)[0]
        
    activation = tf.nn.sigmoid
    # Fully connected layers
    W_fc1 = init_variable([NUM_INPUTS, NUM_HIDDEN])
    b_fc1 = init_variable([NUM_HIDDEN])
    W_fc2 = init_variable([NUM_HIDDEN, NUM_OUTPUTS])
    b_fc2 = init_variable([NUM_OUTPUTS])
    
    def RNN(x,W_fc1,b_fc1,W_fc2,b_fc2):
        # Fully connected layer
        x = tf.reshape(x,[-1,NUM_INPUTS])
        h_fc1 = activation(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1 = tf.reshape(h_fc1, [-1, NUM_TIMESTEPS, NUM_HIDDEN])
        
        # LSTM cell
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN, forget_bias=1.0, state_is_tuple=True)
        
        init_state = lstm_cell.zero_state(SAMPLE_SIZE,dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell,h_fc1, initial_state=init_state, time_major=False)
        
        # Output layer
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
        h_fc2 = activation(tf.matmul(outputs[-1],W_fc2)+b_fc2)

        return h_fc2
        
    y_lstm = RNN(x,W_fc1,b_fc1,W_fc2,b_fc2)
    # cost function
    mse            = tf.reduce_sum(tf.square(y_ - y_lstm))
    regularization = WEIGHT_DECAY*(tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc2))
    loss           = mse+regularization
    
    # optimization setup
    train_step    = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    
    print('Model defined')
    # begin training
    print('Begin training model')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    ex0 = mnist.train.images[0,:]
    ex0 = ex0.reshape((IMAGE_SIZE,IMAGE_SIZE))
    ex0[NUM_TIMESTEPS:,:] = 0 ; 
    for i in range(NUM_STEPS):
        batch_x, _ = mnist.train.next_batch(BATCH_SIZE)
        batch_x, batch_y = gen_training_data(batch_x,NUM_TIMESTEPS)
        train_step.run(feed_dict={x: batch_x, y_: batch_y})
        if i%NUM_STEPS_OUT == 0:         
            mse_batch = mse.eval(feed_dict={x: batch_x, y_: batch_y})
            print("Batch reconstruction MSE (step %d): %g" % (i, mse_batch))
            
            # reconstruct each row
            ex = ex0 ; 
            for k in range(IMAGE_SIZE-NUM_TIMESTEPS):
                ex_new = y_lstm.eval(feed_dict={x: [ex[k:k+NUM_TIMESTEPS,:]]})
                ex[k+NUM_TIMESTEPS] = ex_new
            plt.imshow(ex)
            plt.show()
            
main()