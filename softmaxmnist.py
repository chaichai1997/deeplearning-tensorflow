import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from random import randint
import numpy as np

# 数据准备及相关数据设定
logs_path = 'log_mnist_softmax'
batch_size = 100
learning_rate = 0.5
training_epochs = 10
mnist = input_data.read_data_sets("data", one_hot=True)

# x为28*28的图像
X = tf.placeholder(tf.float32, [None, 784], name="input")
# y为10个元素张量组成的一维数据
Y_ = tf.placeholder(tf.float32, [None, 10])
# 权重矩阵
W = tf.Variable(tf.zeros([784, 10]))
# 偏置
b = tf.Variable(tf.zeros([10]))
# 将图片转化至一维
XX = tf.reshape(X, [-1, 784])

# softmax函数
Y = tf.nn.softmax(tf.matmul(XX, W) + b, name="output")
# 误差，交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
# 计算模型准确率
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

# 定义汇总
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logs_path,
                                   graph=tf.get_default_graph())
    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples/batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, summary = sess.run([train_step, summary_op],
                                  feed_dict={X: batch_x,
                                             Y_: batch_y})
            writer.add_summary(summary, epoch * batch_count + i)
        print("Epoch: ", epoch)

    print("Accuracy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))
    print("done")

    # 在单一图像上运行网络模型
    # 随机选取一张图片
    num = randint(0, mnist.test.images.shape[0])
    img = mnist.test.images[num]
    plt.imshow(img.reshape([28, 28]), cmap='Greys')
    plt.show()
    # 将前面实现的分类器用于选定的图片上
    classification = sess.run(tf.argmax(Y, 1), feed_dict={X: [img]})
    print('Neural Network predicted', classification[0])
    print('Real label is:', np.argmax(mnist.test.labels[num]))
    # 保存模型 .ckpt 保存权重 .ckpt.meta 保存图的定义
    saver = tf.train.Saver()
    save_path = saver.save(sess, "data/saved_mnist_cnn.ckpt")
    print("Model saved to %s" % save_path)