# -*- coding: utf-8 -*-
# author = "chaichai"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data", one_hot=True)
learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
example_to_show = 4
n_hidden_1 = 256   # 隐藏特征
n_hidden_2 = 128
n_input = 784
X = tf.placeholder("float", [None, n_input])
# 定义网络权重和偏差
weights = {   # 对称结构
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}
# 将网络分为两个互补的、全连接的网路，一个编码器和一个解码器
# 将数据编码为一个有用输入
encoder_in = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_h1']), biases['encoder_b1']))
# 编码的第二步，数据的压缩
encoder_out = tf.nn.sigmoid(tf.add(tf.matmul(encoder_in, weights['encoder_h2']), biases['encoder_b2']))
decoder_in = tf.nn.sigmoid(tf.add(tf.matmul(encoder_out, weights['decoder_h1']), biases['decoder_b1']))
decoder_out = tf.nn.sigmoid(tf.add(tf.matmul(decoder_in, weights['decoder_h2']), biases['decoder_b2']))
y_pred = decoder_out
y_true = X
cost = tf.reduce_mean(tf.pow(y_true-y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
# 创建会话
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)  # 设置批图像的大小
    for epoch in range(training_epochs):
        for i in range(total_batch):   # 每个批内进行循环
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})  # 执行优化步骤
        if epoch % display_step == 0:  # 显示每个训练时期的结果
            print("Epoch:", '%04d' % (epoch+1), "cost=", '{:.9f}'.format(c))
    print('Optimization Finished!')
    encoder_decoder = sess.run(y_pred, feed_dict={X: mnist.test.images[:example_to_show]})
    f, a = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(example_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encoder_decoder[i], (28, 28)))
    f.show()
    plt.draw()
    plt.show()
