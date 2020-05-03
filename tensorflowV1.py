import h5py
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from PIL import Image
from scipy import misc
from scipy import ndimage

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

def create_placeholders(nH, nW, nC, nY):
    train = tf.placeholder(tf.float32, [None, nH, nW, nC])
    label = tf.placeholder(tf.float32, [None, nY])
    return train, label

def initialize_parameters():
    tf.set_random_seed(1)
    
    W1 = tf.get_variable('W1', [7, 7, 3, 16], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    b1 = tf.get_variable("b1", [16], initializer=tf.zeros_initializer())
    
    W2 = tf.get_variable('W2', [3, 3, 16, 32], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    b2 = tf.get_variable("b2", [32], initializer=tf.zeros_initializer())
    
    W3 = tf.get_variable('W3', [3, 3, 32, 64], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    b3 = tf.get_variable("b3", [64], initializer=tf.zeros_initializer())
    
    W4 = tf.get_variable('W4', [3, 3, 64, 128], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    b4 = tf.get_variable("b4", [128], initializer=tf.zeros_initializer())
    
    parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2, "W3":W3, "b3":b3, "W4":W4, "b4":b4}
    return parameters

def forward_propagation(X, parameters, rate):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4'] 
    
    #36x36x3
    conv1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME', name = 'conv1')
    #36x36x16
    out1 = tf.nn.bias_add(conv1, b1)
    A1 = tf.nn.relu(out1, name = 'A1')
    P1 = tf.nn.max_pool(A1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'P1')
    #18x18x16
    R1 = tf.nn.lrn(P1, name = 'R1')
    
    conv2 = tf.nn.conv2d(R1, W2, strides = [1,1,1,1], padding = 'SAME', name = 'conv2')
    #18x18x32
    out2 = tf.nn.bias_add(conv2, b2)
    A2 = tf.nn.relu(out2, name = 'A2')
    R2 = tf.nn.lrn(A2, name = 'R2')
    P2 = tf.nn.max_pool(R2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'P2')
    #9x9x32
    
    conv3 = tf.nn.conv2d(P2, W3, strides = [1,1,1,1], padding = 'SAME', name = 'conv3')
    #9x9x64
    out3 = tf.nn.bias_add(conv3, b3)
    A3 = tf.nn.relu(out3, name = 'A3')
    P3 = tf.nn.max_pool(A3, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'P3')
    #4x4x64
    
    conv4 = tf.nn.conv2d(P3, W4, strides = [1,1,1,1], padding = 'SAME', name = 'conv4')
    #4x4x128
    out4 = tf.nn.bias_add(conv4, b4)
    A4 = tf.nn.relu(out4, name = 'A4')
    P4 = tf.nn.max_pool(A4, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'P4')
    #2x2x128
    
    P4_flat = tf.contrib.layers.flatten(P4)
    P4_norm = tf.nn.l2_normalize(P4_flat, axis = -1)
    drop = tf.nn.dropout(P4_norm, keep_prob=1-rate, name = "drop5")
    D1 = tf.layers.dense(drop, 128, activation=tf.nn.relu, name = 'D1', reuse = tf.AUTO_REUSE)
    D2 = tf.layers.dense(D1, 2, name = 'D2', reuse = tf.AUTO_REUSE)
        
    return D2

def test_roger(harry, inputSize = 75):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, nH, nW, nC) = (2*inputSize, 36, 36, 3)
    nY = 2
    X, _ = create_placeholders(nH, nW, nC, nY)
    parameters = initialize_parameters()
    D2 = forward_propagation(X, parameters, 0)
    
    out = tf.argmax(input=D2, axis=1)
        
    saver = tf.train.Saver()
    
    with tf.Session(config = config) as sess:
        
        saver.restore(sess, "./isHarryFaceNonOverfitDropOut.ckpt")
        
        #pos = np.zeros((m))
        #for i in range(0, m):
        pos = out.eval(feed_dict={X:harry})
    return pos
