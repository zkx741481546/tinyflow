import tensorflow as tf
import time
import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

config = tf.ConfigProto()

# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.05
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# config = tf.ConfigProto(gpu_options=gpu_options)


def load_mnist_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('Loading data...')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    return train_set, valid_set, test_set

def convert_to_one_hot(vals):
    one_hot_vals = np.zeros((vals.size, vals.max()+1))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals


# tf.disable_eager_execution()
#
#初始化函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None,10])

#使用1步长stride size，0边距padding size
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

#2*2的max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

#卷积在每个5x5的patch中算出32个特征,卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，
# 接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
x_image=tf.reshape(x,[-1, 28, 28, 1])

# 把x_image和权值向量进行卷积相乘，加上偏置，使用ReLU激活函数，最后maxpooling
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+ b_conv1)
h_pool1=max_pool_2x2(h_conv1)

# 把几个类似的层堆叠起来，第二层中，每个5x5的patch会得到64个特征
W_conv2=weight_variable([5, 5, 32, 64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1, W_conv2)+ b_conv2)
h_pool2=max_pool_2x2(h_conv2)

# 图片降维到7*7，加入一个有1024个神经元的全连接层，用于处理整个图片。
# 把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，使用ReLU激活
W_fc1=weight_variable([7*7*64, 1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1, 7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+ b_fc1)

# 减少过拟合，在输出层之前加入dropout。用一个placeholder来代表一个神经元在dropout中被保留的概率。
# 这样可以在训练过程中启用dropout，在测试过程中关闭dropout。
# TensorFlow的tf.nn.dropout操作会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale。
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1, keep_prob)

# 最后，添加一个softmax层，就像前面的单层softmax回归一样
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
fc2 = tf.matmul(h_fc1_drop, W_fc2)+ b_fc2
y_conv = tf.nn.softmax(fc2)

# cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc2)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

test_y_conv = y_conv

batch_size = 100
X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)

datasets = load_mnist_data("mnist.pkl.gz")
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
print("train_size", train_set_x.shape[0])
print("valid_size", valid_set_x.shape[0])
n_train_batches = train_set_x.shape[0] // batch_size
n_valid_batches = valid_set_x.shape[0] // batch_size




tic = time.time()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    i = 0
    while i < 1000:

        for minibatch_index in range(n_train_batches):
            minibatch_start = minibatch_index * batch_size
            minibatch_end = (minibatch_index + 1) * batch_size
            X_val[:] = train_set_x[minibatch_start:minibatch_end]
            y_val[:] = convert_to_one_hot(train_set_y[minibatch_start:minibatch_end])
            # print("shape",y_val.shape)
            # print(train_set_y[minibatch_start:minibatch_end].shape)
            train_step.run(feed_dict={x: X_val, y_: y_val, keep_prob: 0.5})

            if i%100==0:
                toc = time.time()
                print("use time: " + str(toc - tic))

                train_accuracy = accuracy.eval(feed_dict = {
                    x:X_val, y_: y_val, keep_prob:1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))

                tic = time.time()

            i = i + 1

    #
    # print("test accuracy %g"%accuracy.eval(feed_dict = {
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))

