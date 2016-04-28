# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:24:58 2016

@author: marko
"""

import tensorflow as tf
import numpy as np

path = "/media/marko/9604192C041910BB/Users/Marko/Documents/Visual Studio 2015/Projects/SpiralQ/SpiralQ/Tests/"

target_data = {}
training_data = {}

validation_target_data = {}
validation_training_data = {}

bias_flag = True

for i in range(19):
    training_data[i] = tf.constant(np.reshape(np.fromfile(path+"d_training_data_20_7_314_" + str(i) + ".dat",np.float32),(314,7)).T)
    target_data[i] = tf.constant(np.reshape(np.fromfile(path+"d_target_data_20_7_314_" + str(i) + ".dat",np.float32),(314,7)).T)
    
for i in range(29):
    validation_training_data[i] = tf.constant(np.reshape(np.fromfile(path+"d_training_data_validation_7_35_" + str(i) + ".dat",np.float32),(35,7)).T)
    validation_target_data[i] = tf.constant(np.reshape(np.fromfile(path+"d_target_data_validation_7_35_" + str(i) + ".dat",np.float32),(35,7)).T)
 
class LSTM:
    def __init__(self,input_size,hidden_size):
        
        self.W_z = tf.Variable(tf.random_uniform([hidden_size, input_size], -1.0/np.sqrt(hidden_size+input_size), 1.0/np.sqrt(hidden_size+input_size)))
        self.U_z = tf.Variable(tf.random_uniform([hidden_size, hidden_size], -1.0/np.sqrt(hidden_size+hidden_size), 1.0/np.sqrt(hidden_size+hidden_size)))
        self.b_z = tf.Variable(tf.random_uniform([hidden_size, 1], -1.0/np.sqrt(hidden_size), 1.0/np.sqrt(hidden_size)))        
        
        self.W_i = tf.Variable(tf.random_uniform([hidden_size, input_size], -1.0/np.sqrt(hidden_size+input_size), 1.0/np.sqrt(hidden_size+input_size)))
        self.U_i = tf.Variable(tf.random_uniform([hidden_size, hidden_size], -1.0/np.sqrt(hidden_size+hidden_size), 1.0/np.sqrt(hidden_size+hidden_size)))
        self.b_i = tf.Variable(tf.random_uniform([hidden_size, 1], -1.0/np.sqrt(hidden_size), 1.0/np.sqrt(hidden_size)))     
        self.P_i = tf.Variable(tf.random_uniform([hidden_size, hidden_size], -1.0/np.sqrt(hidden_size+input_size), 1.0/np.sqrt(hidden_size+input_size)))
        
        self.W_f = tf.Variable(tf.random_uniform([hidden_size, input_size], -1.0/np.sqrt(hidden_size+input_size), 1.0/np.sqrt(hidden_size+input_size)))
        self.U_f = tf.Variable(tf.random_uniform([hidden_size, hidden_size], -1.0/np.sqrt(hidden_size+hidden_size), 1.0/np.sqrt(hidden_size+hidden_size)))
        self.b_f = tf.Variable(tf.random_uniform([hidden_size, 1], -1.0/np.sqrt(hidden_size), 1.0/np.sqrt(hidden_size)))     
        self.P_f = tf.Variable(tf.random_uniform([hidden_size, hidden_size], -1.0/np.sqrt(hidden_size+input_size), 1.0/np.sqrt(hidden_size+input_size)))
        
        self.W_o = tf.Variable(tf.random_uniform([hidden_size, input_size], -1.0/np.sqrt(hidden_size+input_size), 1.0/np.sqrt(hidden_size+input_size)))
        self.U_o = tf.Variable(tf.random_uniform([hidden_size, hidden_size], -1.0/np.sqrt(hidden_size+hidden_size), 1.0/np.sqrt(hidden_size+hidden_size)))
        self.b_o = tf.Variable(tf.random_uniform([hidden_size, 1], -1.0/np.sqrt(hidden_size), 1.0/np.sqrt(hidden_size)))     
        self.P_o = tf.Variable(tf.random_uniform([hidden_size, hidden_size], -1.0/np.sqrt(hidden_size+input_size), 1.0/np.sqrt(hidden_size+input_size)))
        
        self.block_input_a = tf.tanh
        self.block_output_a = tf.tanh
        
    def runLayer (self,x,y,c):
        if bias_flag == True:
            block_input = self.block_input_a(tf.matmul(self.W_z,x) + tf.matmul(self.U_z,y) + self.b_z)
            input_gate = tf.sigmoid(tf.matmul(self.W_i,x) + tf.matmul(self.U_i,y) + tf.matmul(self.P_i,c) + self.b_i)
            forget_gate = tf.sigmoid(tf.matmul(self.W_f,x) + tf.matmul(self.U_f,y) + tf.matmul(self.P_f,c) + self.b_f)
            c_new = block_input*input_gate+c*forget_gate
            output_gate = tf.sigmoid(tf.matmul(self.W_o,x) + tf.matmul(self.U_o,y) + tf.matmul(self.P_o,c_new) + self.b_o)
            return self.block_output_a(c_new)*output_gate, c_new
        else:
            block_input = self.block_input_a(tf.matmul(self.W_z,x) + tf.matmul(self.U_z,y))
            input_gate = tf.sigmoid(tf.matmul(self.W_i,x) + tf.matmul(self.U_i,y) + tf.matmul(self.P_i,c))
            forget_gate = tf.sigmoid(tf.matmul(self.W_f,x) + tf.matmul(self.U_f,y) + tf.matmul(self.P_f,c))
            c_new = block_input*input_gate+c*forget_gate
            output_gate = tf.sigmoid(tf.matmul(self.W_o,x) + tf.matmul(self.U_o,y) + tf.matmul(self.P_o,c_new))
            return self.block_output_a(c_new)*output_gate, c_new

    def runLayerNoH (self,x):
        if bias_flag == True:
            #print(self.W_z.get_shape(),x)
            block_input = self.block_input_a(tf.matmul(self.W_z,x) + self.b_z)
            input_gate = tf.sigmoid(tf.matmul(self.W_i,x) + self.b_i)
            forget_gate = tf.sigmoid(tf.matmul(self.W_f,x) + self.b_f)
            c_new = block_input*input_gate
            output_gate = tf.sigmoid(tf.matmul(self.W_o,x) + self.b_o)
            return self.block_output_a(c_new)*output_gate, c_new
        else:
            block_input = self.block_input_a(tf.matmul(self.W_z,x))
            input_gate = tf.sigmoid(tf.matmul(self.W_i,x))
            forget_gate = tf.sigmoid(tf.matmul(self.W_f,x))
            c_new = block_input*input_gate
            output_gate = tf.sigmoid(tf.matmul(self.W_o,x))
            return self.block_output_a(c_new)*output_gate, c_new

input_size = 7
hidden_size = 128
        
l1 = {1 : LSTM(input_size,hidden_size), 2 : LSTM(hidden_size,hidden_size), 3 : LSTM(hidden_size,hidden_size), 4 : LSTM(hidden_size,hidden_size), 5 : LSTM(hidden_size,hidden_size)}

l2_W = tf.Variable(tf.random_uniform([input_size, hidden_size], -1.0/np.sqrt(hidden_size+input_size), 1.0/np.sqrt(hidden_size+input_size)))
l2_b = tf.Variable(tf.random_uniform([input_size, 1], -1.0/np.sqrt(hidden_size), 1.0/np.sqrt(hidden_size)))


def iterate_time(num_iters,training_data,target_data,batch_size):
    h = {}
    
    for i in range(num_iters):
        h[i,0] = training_data[i], None
    for (k,lstm) in l1.viewitems():
        hid,cell = h[0,k-1]
        h[0,k] = lstm.runLayerNoH(hid)
    
    hid,cell = h[0,len(l1)]
    y = tf.clip_by_value(tf.sigmoid(tf.matmul(l2_W,hid) + l2_b),0.0001,0.9999)
    c = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y,target_data[0])))
    
    for i in range(1,num_iters):
        for (k,lstm) in l1.viewitems():
            hid,cell = h[i,k-1]
            prev_hid,prev_cell = h[i-1,k]
            h[i,k] = lstm.runLayer(hid,prev_hid,prev_cell)
        hid,cell = h[i,len(l1)]
        y = tf.clip_by_value(tf.sigmoid(tf.matmul(l2_W,hid) + l2_b),0.0001,0.9999)
        c += tf.reduce_mean(tf.reduce_sum( tf.squared_difference(y,target_data[i])))
    
    return c / (2.0 * batch_size * num_iters)
    
z = iterate_time(19,training_data,target_data,314)
z_val = iterate_time(29,validation_training_data,validation_target_data,35)
            
train_step = tf.train.GradientDescentOptimizer(15.0).minimize(z)
init = tf.initialize_all_variables()

num_iterations = 100

with tf.Session() as sess:
    sess.run(init)
        
    print ("The cost on the validation set is " + str(z_val.eval()) + " on iteration 0")
    print ("The cost on the training set is " + str(z.eval()) + " on iteration 0")
    sess.run(train_step)

    from timeit import default_timer as timer

    start = timer()    
    
    for i in xrange(1,num_iterations):
        #print ("The cost on the validation set is " + str(z_val.eval()) + " on iteration " + str(i))
        #print ("The cost on the training set is " + str(z.eval()) + " on iteration " + str(i))
        print(i)
        sess.run(train_step)
        
    end = timer()
    
    print ("The cost on the training set is " + str(z.eval()) + " on iteration " + str(i))
    
    print ("The time elapsed is ", end-start)
