# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 01:02:42 2020

@author: Administrator
"""

import tensorflow as tf

#-----------------section 1---------------------------------------
#use of Tensor
#define variable
a = tf.compat.v1.constant([1, 2, 3], name = 'a')
b = tf.constant([[1, 2, 3],
                 [4, 5, 6]])

print(a)
print(b)

#create array
c = tf.zeros([2, 3])
d = tf.ones([2, 3])
print(c)
print(d)

#randomly generate a normal distribution
e = tf.random.normal([5, 3])
print(e)

#-----------------section 2---------------------------------------
#use of session
#generate two matrixes
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])

product = tf.matmul(matrix1, matrix2)

#first
sess= tf.compat.v1.Session()
output = sess.run(product)
print(output)
sess.close()

#second
with tf.compat.v1.Session() as sess:
    output = sess.run(product)
    print(output)
    
#-----------------section 3---------------------------------------
#constant & variable

#generate two constants
c1 = tf.constant([[1, 1]])
c2 = tf.constant([[2], [2]])

state = tf.Variable(0, name = 'counter')

#must use a tensor to be the parameter of Variable
w = tf.Variable(tf.random.normal([10, 20], stddev = 0.35), name = 'w')

one = tf.constant(1)
result = tf.add(state, one)
update = tf.compat.v1.assign(state, result)

#must have if define variable
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    
    for _in in range(3):
        sess.run(update)
        print(sess.run(state))
        
#-----------section 4-----------------------------------------
#placeholder

input1 = tf.compat.v1.placeholder(tf.float32)
input2 = tf.compat.v1.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.compat.v1.Session() as sess:
    print(sess.run(output, feed_dict= {input1:[7.], input2:[8.]}))
    
#------------section 5---------------------------------------
aa = tf.constant([-1.0, 2.0])

with tf.compat.v1.Session() as sess:
    bb = tf.nn.relu(aa)
    print(sess.run(bb))
    
    cc = tf.sigmoid(aa)
    print(sess.run(cc))
