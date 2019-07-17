# -*- coding: utf-8 -*-
import tensorflow as tf
# 初始化模型参数
w = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")

# 将输入值进行合并


def combine(x):
    return tf.matmul(x, w)+b


# sigmoid 函数，输入值为前面合并值


def interfence(x):
    return tf.sigmoid(combine(x))

# 损失函数:交叉熵


def loss(x, y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(combine(x),y))



