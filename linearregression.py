# -*- coding: utf-8 -*-
import tensorflow as tf
# 初始化模型参数
w = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0, name="bias")


# 推断模型再数据x上的输出

def interfence(x):
    return tf.matmul(x, w)+b


# 嵌入输入数据

def input():
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30],
                  [76, 67], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44],
                  [27, 24], [89, 31]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288,
                         385, 402, 365, 209, 290]
    return tf.to_float(weight_age),tf.to_float(blood_fat_content)

# 损失函数

def loss(x,y):
    y_predicted = interfence(x)
    return tf.reduce_sum(tf.squared_difference(y, y_predicted))

# 训练
def train():
    learning_rate = 0.0000001
    return  tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
