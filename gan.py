#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generative adversarial neural network example using MNIST


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

# GAN architecture parameters
NUM_INPUTS      = 100
NUM_HIDDEN      = 128


# learning parameters
BATCH_SIZE      = 100
NUM_STEPS       = 20000
NUM_STEPS_OUT   = 1000 # display accuracy every NUM_STEPS_OUT steps
LEARNING_RATE_G = 1e-3
LEARNING_RATE_D = 1e-3
WEIGHT_DECAY    = 0   # regularization

def init_variable(shape):
    input_dim = shape[0]
    sigma  = tf.sqrt(1./input_dim)
    return tf.Variable(tf.random_normal(shape=shape, stddev=sigma))

def main():
    # load MNIST dataset
    print('Loading data...')
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print('Data loaded')
    
    print('Defining model architecture')
    # input/output placeholders
    x = tf.placeholder(tf.float32, shape=[None, NUM_PIXELS]) # images
    z = tf.placeholder(tf.float32, shape=[None, NUM_INPUTS]) # random code
    
    activation     = tf.nn.relu
    activation_out = tf.nn.sigmoid
    
    ### Generator
    # model parameters
    G_W_fc1 = init_variable([NUM_INPUTS, NUM_HIDDEN])
    G_b_fc1 = init_variable([NUM_HIDDEN])
    G_W_fc2 = init_variable([NUM_HIDDEN, NUM_PIXELS])
    G_b_fc2 = init_variable([NUM_PIXELS])
        
    # generate samples
    def generator(z):
        G_h_fc1 = activation(tf.matmul(z, G_W_fc1) + G_b_fc1)    
        G_h_fc2 = tf.matmul(G_h_fc1, G_W_fc2) + G_b_fc2   
        G_out   = activation_out(G_h_fc2)
    
        return G_out
    
    ### Discriminator
    # model parameters
    D_W_fc1 = init_variable([NUM_PIXELS, NUM_HIDDEN])
    D_b_fc1 = init_variable([NUM_HIDDEN])
    D_W_fc2 = init_variable([NUM_HIDDEN, 1])
    D_b_fc2 = init_variable([NUM_HIDDEN])
    
    # evaluate discriminator
    def discriminator(x):
        D_h_fc1 = activation(tf.matmul(x, D_W_fc1) + D_b_fc1)
        D_h_fc2 = tf.matmul(D_h_fc1, D_W_fc2) + D_b_fc2    
        D_out   = D_h_fc2
        
        return D_out
    
    # generate fake samples
    G_fake = generator(z)
    # evaluate discriminator for real and fake data
    D_fake = discriminator(G_fake)
    D_real = discriminator(x)

    # cost functions for different networks
    # binary classification (0 for fake, 1 for real) --> sigmoid with cross entropy
    # Generator: make discriminator think fake is real!
    G_loss      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels= tf.ones_like(D_fake)))
    
    # Discriminator: standard loss function
    D_loss0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
    D_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels= tf.ones_like(D_real)))
    D_loss  = D_loss0 + D_loss1
    
    # optimization setup: have generator learn faster than discriminator
    G_train_step = tf.train.AdamOptimizer(LEARNING_RATE_G).minimize(G_loss, var_list=[G_W_fc1,G_b_fc1,G_W_fc2,G_b_fc2])
    D_train_step = tf.train.AdamOptimizer(LEARNING_RATE_D).minimize(D_loss, var_list=[D_W_fc1,D_b_fc1,D_W_fc2,D_b_fc2])
        
    print('Model defined')
    
    # begin training
    print('Begin training model')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    for i in range(NUM_STEPS):
        batch    = mnist.train.next_batch(BATCH_SIZE)[0]
        batch_nz = np.random.uniform(-1.,1., size=[BATCH_SIZE,NUM_INPUTS])
        
        G_train_step.run(feed_dict={z: batch_nz}) ;                
        D_train_step.run(feed_dict={x: batch, z: batch_nz})

        if i%NUM_STEPS_OUT == 0:      
            g_loss, d_loss = sess.run([G_loss, D_loss],feed_dict={x: batch, z: batch_nz})
            print("Generative loss, discriminator loss (step %d): %g, %g" % (i, g_loss, d_loss))               

            z_fake = G_fake.eval(feed_dict={z: np.random.uniform(-1.,1., size=[1,NUM_INPUTS])})
            plt.imshow(np.reshape(z_fake, (28,28)))
            plt.show()

main()