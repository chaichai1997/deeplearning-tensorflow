# -*- coding: utf-8 -*-
import tensorflow as tf

a = tf.constant(5, name='input_a')   # 定义一个Operation/节点
b = tf.constant(3, name="input_b")
c = tf.multiply(a, b, name='mul_c')
d = tf.add(a, b, name="add_d")
e = tf.add(c, d, name="add_e")

sess = tf.Session()
print(sess.run(e))
sess.close()
import numpy as np
s = np.array([2, 3])
print(s)