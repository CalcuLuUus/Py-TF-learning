# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 02:36:00 2020

@author: Administrator
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 下载数据集 数字1到10
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# mnist=tf.keras.datasets.mnist

# (x_, y_), (x__, y__) = mnist.load_data()
# print(x__)
# print(y__)

#-----define layer---------

def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.compat.v1.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b= tf.matmul(inputs, Weights) + biases
    
    if activation_function == None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#-----define compute_accuracy-------
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result

#-----placeholder------------

xs = tf.compat.v1.placeholder(tf.float32, [None, 784])
ys = tf.compat.v1.placeholder(tf.float32, [None, 10])

#-----define prediction-----------

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

#-----define loss & train----------
cross_entropyloss = tf.reduce_mean(-tf.compat.v1.reduce_sum(ys * tf.compat.v1.log(prediction), 
                     reduction_indices=[1]))
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropyloss)


#-----session-------------
sess = tf.compat.v1.Session()

init = tf.compat.v1.initialize_all_variables()
sess.run(init)
print(mnist)
for i in range(1000):
    # (batch_xs, batch_ys), (x___, y___) = mnist.load_data() #get 100 sample
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))

