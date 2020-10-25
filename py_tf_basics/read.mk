主要讲了tensor Session Variable constant placeholder activationfunction

# tensor
张量，可以看成多维向量

# session
我们知道，tf是以数据流图的形式运行的，每一个节点是一个operation，每一条边是tensor，从而构成一张图
在构建阶段，operation的步骤会被描述成一张图
执行阶段我们需要用session执行operation，operation只能发生在sessino中

所以对于每一个op的运行，我们都需要用sess

#variable
创建变量需要有tensor作为初始化函数的参数，并且变量一定需要用tf.Variable()来初始化
然后，如果有变量，定义session必须要sess.run(init) [init = tf.compat.v1.global_variables_initializer()]

#placeholder
占位置

#activation function
激活函数
relu sigmoid
