# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 00:50:26 2020

@author: Administrator
"""

import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

#---------define layer----------------
def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.compat.v1.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
        
    return outputs
    
#-----generate data--------
x_data = np.linspace(-1, 1, 300)[:,np.newaxis]

noise = np.random.normal(0, 0.05, x_data.shape)

y_data = np.square(x_data) - 0.5 + noise

xs = tf.compat.v1.placeholder(tf.float32, [None, 1])
ys = tf.compat.v1.placeholder(tf.float32, [None, 1])

#-----look------

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# 散点图
ax.scatter(x_data, y_data)
# 连续显示
plt.ion()
plt.show()

#------define a neural network-------
L1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)

prediction = add_layer(L1, 10, 1, activation_function=None)

#---------define loss---------
loss = tf.reduce_mean(tf.compat.v1.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.compat.v1.initialize_all_variables()

sess = tf.compat.v1.Session()
sess.run(init)

#------nn learning----------------
n = 1
for i in range(1000):
    # 训练
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data}) #假设用全部数据x_data进行运算
    # 输出结果 只要通过place_holder运行就要传入参数
    if i % 50==0:
        #print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        try:
            # 忽略第一次错误 后续移除lines的第一个线段
            ax.lines.remove(lines[0])
        except Exception:
            pass
        
        # 预测
        prediction_value = sess.run(prediction, feed_dict={xs:x_data})
        # 设置线宽度为5 红色
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5) 
        # 暂停
        plt.pause(0.1)
        # 保存图片
        name = "test" + str(n) + ".png"
        #plt.savefig(name)
        n =  n + 1
