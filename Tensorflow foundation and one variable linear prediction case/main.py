# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:04:43 2020

@author: Administrator
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
x_data = np.random.rand(100).astype(np.float32)
# print(x_data)
y_data = x_data * 0.1 + 0.3
print(y_data)

Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
# print(Weights)
biases = tf.Variable(tf.zeros([1]))
# print(biases)

y = Weights * x_data + biases
loss = tf.compat.v1.reduce_mean(tf.square(y-y_data))


optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(loss)

init = tf.compat.v1.global_variables_initializer()

# 定义Session 
sess = tf.compat.v1.Session()
# 运行时Session就像一个指针 指向要处理的位置并激活
sess.run(init)  
# 训练201次
for n in range(201):
    # 训练并且每隔20次输出结果
    sess.run(train)
    if n % 20 == 0:
        # 最佳拟合结果 W: [0.100], b: [0.300]
        print(n, sess.run(Weights), sess.run(biases))
        pre = x_data * sess.run(Weights) + sess.run(biases)

print("pre")
print(pre)
plt.scatter(x_data, y_data)
plt.plot(x_data, pre, 'r-')
plt.show()

