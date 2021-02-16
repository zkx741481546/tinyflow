import six.moves.cPickle as pickle
import gzip
import os
from pycode.tinyflow import autodiff as ad
from pycode.tinyflow import ndarray
from pycode.tinyflow import gpu_op
from pycode.tinyflow import TrainExecute
import numpy as np
import random
from pycode.tinyflow import train
import pickle
import time


class_num = 10
image_size = 32
img_channels = 3
iterations = 200
batch_size = 100
weight_decay = 0.0003
dropout_rate = 0.5
momentum_rate = 0.9
data_dir = './cifar-10/'
log_save_path = './vgg_16_logs'
model_save_path = './model/'

# 读文件
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


# 从读入的文件中获取图片数据(data)和标签信息(labels)
def load_data_one(file):
    batch = unpickle(file)
    data = batch['data']
    labels = batch['labels']
    print("Loading %s : img num %d." % (file, len(data)))
    return data, labels


# 将从文件中获取的信息进行处理，得到可以输入到神经网络中的数据。
def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)

    # 标签labels从0-9的数字转化为float类型(-1,10)的标签矩阵
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    # 将图片数据从(-1,3072)转化为(-1,3,32,32)
    data = data.reshape([-1, img_channels, image_size, image_size])
    # 将(-1,3,32,32)转化为(-1,32,32,3)的图片标准输入
    # data = data.transpose([0, 2, 3, 1])

    # data数据归一化
    data = data.astype('float32')
    data[:, :, :, 0] = (data[:, :, :, 0] - np.mean(data[:, :, :, 0])) / np.std(data[:, :, :, 0])
    data[:, :, :, 1] = (data[:, :, :, 1] - np.mean(data[:, :, :, 1])) / np.std(data[:, :, :, 1])
    data[:, :, :, 2] = (data[:, :, :, 2] - np.mean(data[:, :, :, 2])) / np.std(data[:, :, :, 2])

    return data, labels


def prepare_data():
    print("======Loading data======")
    image_dim = image_size * image_size * img_channels
    meta = unpickle(data_dir + 'batches.meta')
    print(meta)

    label_names = meta['label_names']

    # 依次读取data_batch_1-5的内容
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, class_num)
    test_data, test_labels = load_data(['test_batch'], data_dir, class_num)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    # 重新打乱训练集的顺序
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======数据准备结束======")

    return train_data, train_labels, test_data, test_labels




def sgd_update_gpu(param, grad_param, learning_rate):
    """Helper GPU SGD update method. Avoids copying NDArray to cpu."""
    assert isinstance(param, ndarray.NDArray)
    assert isinstance(grad_param, ndarray.NDArray)
    gpu_op.matrix_elementwise_multiply_by_const(
        grad_param, -learning_rate, grad_param)
    gpu_op.matrix_elementwise_add(param, grad_param, param)

def convert_to_one_hot(vals):
    """Helper method to convert label array to one-hot array."""
    one_hot_vals = np.zeros((vals.size, vals.max()+1))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals

def map_to_numpy(map):
    list = []
    i = 0
    for node, value in map.items():
        list.append(i)
        list.append(value.asnumpy())
    # del (list[0])
    return list


def vgg16(num_step = 1000):

    n_class = 10

    X = ad.Placeholder("inputs")
    y_ = ad.Placeholder("y_")
    filters1_1 = ad.Variable("filters1_1")
    filters1_2 = ad.Variable("filters1_2")
    filters2_1 = ad.Variable("filters2_1")
    filters2_2 = ad.Variable("filters2_2")
    filters3_1 = ad.Variable("filters3_1")
    filters3_2 = ad.Variable("filters3_2")
    filters3_3 = ad.Variable("filters3_3")
    filters4_1 = ad.Variable("filters4_1")
    filters4_2 = ad.Variable("filters4_2")
    filters4_3 = ad.Variable("filters4_3")
    filters5_1 = ad.Variable("filters5_1")
    filters5_2 = ad.Variable("filters5_2")
    filters5_3 = ad.Variable("filters5_3")
    filters6 = ad.Variable("filters6")
    filters7 = ad.Variable("filters7")
    filters8 = ad.Variable("filters8")
    b1_1 = ad.Variable("b1_1")
    b1_2 = ad.Variable("b1_2")
    b2_1 = ad.Variable("b2_1")
    b2_2 = ad.Variable("b2_2")
    b3_1 = ad.Variable("b3_1")
    b3_2 = ad.Variable("b3_2")
    b3_3 = ad.Variable("b3_3")
    b4_1 = ad.Variable("b4_1")
    b4_2 = ad.Variable("b4_2")
    b4_3 = ad.Variable("b4_3")
    b5_1 = ad.Variable("b5_1")
    b5_2 = ad.Variable("b5_2")
    b5_3 = ad.Variable("b5_3")
    b6 = ad.Variable("b6")
    b7 = ad.Variable("b7")
    b8 = ad.Variable("b8")




    # conv 1
    conv1_1 = ad.conv2withbias(X, filters1_1, b1_1, "NCHW", "SAME", 1, 1)
    bn1_1 = ad.bn_forward_op(conv1_1, "NCHW", "pre_activation")
    act1_1 = ad.activation_forward_op(bn1_1, "NCHW", "relu")

    conv1_2 = ad.conv2withbias(act1_1, filters1_2, b1_2, "NCHW", "SAME", 1, 1)
    bn1_2 = ad.bn_forward_op(conv1_2, "NCHW", "pre_activation")
    act1_2 = ad.activation_forward_op(bn1_2, "NCHW", "relu")
    pool1 = ad.pooling_2d_forward_op(act1_2, "NCHW", "max", 0, 0, 2, 2, 2, 2)

    # conv 2
    conv2_1 = ad.conv2withbias(pool1, filters2_1, b2_1, "NCHW", "SAME", 1, 1)
    bn2_1 = ad.bn_forward_op(conv2_1, "NCHW", "pre_activation")
    act2_1 = ad.activation_forward_op(bn2_1, "NCHW", "relu")
    conv2_2 = ad.conv2withbias(act2_1, filters2_2, b2_2, "NCHW", "SAME", 1, 1)
    bn2_2 = ad.bn_forward_op(conv2_2, "NCHW", "pre_activation")
    act2_2 = ad.activation_forward_op(bn2_2, "NCHW", "relu")
    pool2 = ad.pooling_2d_forward_op(act2_2, "NCHW", "max", 0, 0, 2, 2, 2, 2)

    # conv 3
    conv3_1 = ad.conv2withbias(pool2, filters3_1, b3_1, "NCHW", "SAME", 1, 1)
    bn3_1 = ad.bn_forward_op(conv3_1, "NCHW", "pre_activation")
    act3_1 = ad.activation_forward_op(bn3_1, "NCHW", "relu")
    conv3_2 = ad.conv2withbias(act3_1, filters3_2, b3_2,"NCHW", "SAME", 1, 1)
    bn3_2 = ad.bn_forward_op(conv3_2, "NCHW", "pre_activation")
    act3_2 = ad.activation_forward_op(bn3_2, "NCHW", "relu")
    conv3_3 = ad.conv2withbias(act3_2, filters3_3, b3_3, "NCHW", "SAME", 1, 1)
    bn3_3 = ad.bn_forward_op(conv3_3, "NCHW", "pre_activation")
    act3_3 = ad.activation_forward_op(bn3_3, "NCHW", "relu")
    pool3 = ad.pooling_2d_forward_op(act3_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

    # conv 4
    conv4_1 = ad.conv2withbias(pool3, filters4_1, b4_1, "NCHW", "SAME", 1, 1)
    bn4_1 = ad.bn_forward_op(conv4_1, "NCHW", "pre_activation")
    act4_1 = ad.activation_forward_op(bn4_1, "NCHW", "relu")
    conv4_2 = ad.conv2withbias(act4_1, filters4_2, b4_2, "NCHW", "SAME", 1, 1)
    bn4_2 = ad.bn_forward_op(conv4_2, "NCHW", "pre_activation")
    act4_2 = ad.activation_forward_op(bn4_2, "NCHW", "relu")
    conv4_3 = ad.conv2withbias(act4_2, filters4_3, b4_3, "NCHW", "SAME", 1, 1)
    bn4_3 = ad.bn_forward_op(conv4_3, "NCHW", "pre_activation")
    act4_3 = ad.activation_forward_op(bn4_3, "NCHW", "relu")
    pool4 = ad.pooling_2d_forward_op(act4_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

    # conv 5
    conv5_1 = ad.conv2withbias(pool4, filters5_1, b5_1, "NCHW", "SAME", 1, 1)
    bn5_1 = ad.bn_forward_op(conv5_1, "NCHW", "pre_activation")
    act5_1 = ad.activation_forward_op(bn5_1, "NCHW", "relu")
    conv5_2 = ad.conv2withbias(act5_1, filters5_2, b5_2, "NCHW", "SAME", 1, 1)
    bn5_2 = ad.bn_forward_op(conv5_2, "NCHW", "pre_activation")
    act5_2 = ad.activation_forward_op(bn5_2, "NCHW", "relu")
    conv5_3 = ad.conv2withbias(act5_2, filters5_3, b5_3, "NCHW", "SAME", 1, 1)
    bn5_3 = ad.bn_forward_op(conv5_3, "NCHW", "pre_activation")
    act5_3 = ad.activation_forward_op(bn5_3, "NCHW", "relu")
    pool5 = ad.pooling_2d_forward_op(act5_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

    # fc6
    pool5_flat = ad.flatten_op(pool5)
    fc6 = ad.dense(pool5_flat, filters6, b6)
    bn6 = ad.fullybn_forward_op(fc6, "NCHW")
    act6 = ad.fullyactivation_forward_op(bn6, "NCHW", "relu")
    drop6 = ad.fullydropout_forward_op(act6, "NCHW", 0.5)

    # fc7
    fc7 = ad.dense(drop6, filters7, b7)
    bn7 = ad.fullybn_forward_op(fc7, "NCHW")
    act7 = ad.fullyactivation_forward_op(bn7, "NCHW", "relu")
    drop7 = ad.fullydropout_forward_op(act7, "NCHW", 0.5)

    #fc8
    fc8 = ad.dense(drop7, filters8, b8)
    bn8 = ad.fullybn_forward_op(fc8, "NCHW")
    act8 = ad.fullyactivation_forward_op(bn8, "NCHW", "softmax")

    loss = ad.softmaxcrossentropy_op(bn8, y_)

    # grad = ad.gradients(loss, [filters1_1, filters1_2, filters2_1, filters2_2, filters3_1, filters3_2, filters3_3
    #                             , filters4_1, filters4_2, filters4_3, filters5_1, filters5_2, filters5_3
    #                             , filters6, filters7])
    # executor = ad.Executor([grad[0], grad[1], grad[2], grad[3], grad[4], grad[5], grad[6], grad[7], grad[8], grad[9]
    #                            , grad[10], grad[11], grad[12], grad[13], grad[14], loss, y_], ctx=ctx)

    train_x, train_y, test_x, test_y = prepare_data()
    n_train_batches = train_x.shape[0] // batch_size
    n_test_batches = test_x.shape[0] // batch_size


    X_val = np.empty(shape=(batch_size, 3, 32, 32), dtype=np.float32)
    y_val = np.empty(shape=(batch_size, n_class), dtype=np.float32)
    filters_val = [np.random.normal(0.0, 0.1, (64, 3, 3, 3))]
    filters_val.append(np.random.normal(0.0, 0.1, (64, 64, 3, 3)))
    filters_val.append(np.random.normal(0.0, 0.1, (128, 64, 3, 3)))
    filters_val.append(np.random.normal(0.0, 0.1, (128, 128, 3, 3)))
    filters_val.append(np.random.normal(0.0, 0.1, (256, 128, 3, 3)))
    filters_val.append(np.random.normal(0.0, 0.1, (256, 256, 3, 3)))
    filters_val.append(np.random.normal(0.0, 0.1, (256, 256, 3, 3)))
    filters_val.append(np.random.normal(0.0, 0.1, (512, 256, 3, 3)))
    filters_val.append(np.random.normal(0.0, 0.1, (512, 512, 3, 3)))
    filters_val.append(np.random.normal(0.0, 0.1, (512, 512, 3, 3)))
    filters_val.append(np.random.normal(0.0, 0.1, (512, 512, 3, 3)))
    filters_val.append(np.random.normal(0.0, 0.1, (512, 512, 3, 3)))
    filters_val.append(np.random.normal(0.0, 0.1, (512, 512, 3, 3)))
    filters_val.append(np.random.normal(0.0, 0.1, (512 * 1 * 1, 4096)))
    filters_val.append(np.random.normal(0.0, 0.1, (4096, 4096)) * 0.001)
    filters_val.append(np.random.normal(0.0, 0.1, (4096, n_class)) * 0.001)
    b_val = [np.ones(64) * 0.1]
    b_val.append(np.ones(64) * 0.1)
    b_val.append(np.ones(128) * 0.1)
    b_val.append(np.ones(128) * 0.1)
    b_val.append(np.ones(256) * 0.1)
    b_val.append(np.ones(256) * 0.1)
    b_val.append(np.ones(256) * 0.1)
    b_val.append(np.ones(512) * 0.1)
    b_val.append(np.ones(512) * 0.1)
    b_val.append(np.ones(512) * 0.1)
    b_val.append(np.ones(512) * 0.1)
    b_val.append(np.ones(512) * 0.1)
    b_val.append(np.ones(512) * 0.1)
    b_val.append(np.ones(4096) * 0.1)
    b_val.append(np.ones(4096) * 0.1)
    b_val.append(np.ones(n_class) * 0.1)

    # ctx = ndarray.gpu(0)
    # for i in range(16):
    #     filters_val[i] = ndarray.array(filters_val[i], ctx)
    # for i in range(16):
    #     b_val[i] = ndarray.array(b_val[i], ctx)

    aph = 0.0001
    t = TrainExecute.TrainExecutor(loss, aph)
    t.init_Variable(
        {filters1_1: filters_val[0], filters1_2: filters_val[1], filters2_1: filters_val[2], filters2_2: filters_val[3]
            , filters3_1: filters_val[4], filters3_2: filters_val[5], filters3_3: filters_val[6]
            , filters4_1: filters_val[7], filters4_2: filters_val[8], filters4_3: filters_val[9]
            , filters5_1: filters_val[10], filters5_2: filters_val[11], filters5_3: filters_val[12]
            , filters6: filters_val[13], filters7: filters_val[14], filters8: filters_val[15]
            , b1_1: b_val[0], b1_2: b_val[1], b2_1: b_val[2], b2_2: b_val[3], b3_1: b_val[4]
            , b3_2: b_val[5], b3_3: b_val[6], b4_1: b_val[7], b4_2: b_val[8], b4_3: b_val[9]
            , b5_1: b_val[10], b5_2: b_val[11], b5_3: b_val[12]
            , b6: b_val[13], b7: b_val[14], b8: b_val[15]})


    for i in range(num_step):
        print("step %d" % i)
        X_val = np.empty(shape=(0, 3, 32, 32), dtype=np.float32)
        y_val = np.empty(shape=(0, n_class), dtype=np.float32)

        for j in range(batch_size):
            select = random.randint(0, n_train_batches)
            temp_train_x = train_x[select]
            temp_train_x = temp_train_x[np.newaxis, :]
            temp_train_y = train_y[select]
            temp_train_y = temp_train_y[np.newaxis, :]
            X_val = np.append(X_val, temp_train_x, axis=0)
            y_val = np.append(y_val, temp_train_y, axis=0)
        # print("train", X_val.shape)
        # print("train", y_val.shape)
        t.run({X: X_val, y_: y_val})

        if (i+1) % 100 == 0 :
            X_test_val = np.empty(shape=(100, 3, 32, 32), dtype=np.float32)
            y_test_val = np.empty(shape=(100, n_class), dtype=np.float32)
            correct_predictions = []
            for minibatch_index in range(n_test_batches):
                minibatch_start = minibatch_index * batch_size
                minibatch_end = (minibatch_index + 1) * batch_size
                X_test_val[:] = test_x[minibatch_start:minibatch_end]
                y_test_val[:] = test_y[minibatch_start:minibatch_end]
                # print(X_test_val.shape)
                # print(y_test_val.shape)
                feed_dict = {X: X_test_val, y_: y_test_val}
                dict_Variable = t.get_Variable_node_to_val_map()
                feed_dict.update(dict_Variable)
                y_predicted = t.run_get_nodelist_once(feed_dict, [act8])[act8].asnumpy()
                correct_prediction = np.equal(
                    np.argmax(y_test_val, 1),
                    np.argmax(y_predicted, 1)).astype(np.float)
                correct_predictions.extend(correct_prediction)
            accuracy = np.mean(correct_predictions)
            print("validation set accuracy=%f" % accuracy)

    return filters_val



vgg16(1000)