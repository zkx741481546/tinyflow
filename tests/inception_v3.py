from pycode.tinyflow import autodiff as ad
import numpy as np
from pycode.tinyflow import ndarray
from pycode.tinyflow import gpu_op
from pycode.tinyflow import train
import pickle
class_num = 10
image_size = 32
img_channels = 3
iterations = 200
batch_size = 250
weight_decay = 0.0003
dropout_rate = 0.5
momentum_rate = 0.9
data_dir = './cifar-10/'
log_save_path = './vgg_16_logs'
model_save_path = './model/'
n_class = 10
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
def test_concat():
    inputs1 = ad.Variable("inputs1")
    inputs2 = ad.Variable("inputs1")

    # ini
    ctx = ndarray.gpu(0)
    x_val1 = np.linspace(2, 20, 10).reshape((5,2))*0.5
    x_val1 = ndarray.array(x_val1, ctx=ctx)
    print(x_val1)
    x_val2 = np.ones((5,3))*0.1
    x_val2= ndarray.array(x_val2, ctx=ctx)
    loss = ad.concat_forward_op(inputs1,inputs2)
    # loss = ad.l1regular_op(inputs)
    grad_f = ad.gradients(loss, [inputs1,inputs2])  # gra返回一个list
    executor = ad.Executor([loss,grad_f[0],grad_f[1]], ctx=ctx)
    g_val = executor.run(feed_dict={inputs1: x_val1,inputs2:x_val2})  # 返回一个list
    print("g_val:", g_val[0].asnumpy())
    print("g_val:", g_val[1].asnumpy())
    print("g_val:", g_val[2].asnumpy())
def inception_v3():
    X = ad.Placeholder("inputs")
    y_ = ad.Placeholder("y_")
    inputs = ad.Variable("inputs")
    filterb_1 = ad.Variable("filterb_1")
    filterb_2 = ad.Variable("filterb_2")
    filterb_3 = ad.Variable("filterb_3")
    filterb_4 = ad.Variable("filterb_4")
    filterb_5= ad.Variable("filterb_5")

    ctx = ndarray.gpu(0)
    filtersb_val1 = np.ones((32,3,3,3)) * 0.01
    filtersb_val2 = np.ones((32,32,3,3)) * 0.01
    filtersb_val3 = np.ones((64,32,3,3)) * 0.01
    filtersb_val4 = np.ones((80, 64, 1, 1)) * 0.01
    filtersb_val5 = np.ones((192, 80, 3, 3)) * 0.01
    y_val = np.zeros((147,147,64))

#inception前
    covb_1=ad.convolution_2d_forward_op(inputs, filterb_1, "NCHW", "VALID", 2, 2)
    covb_2=ad.convolution_2d_forward_op(covb_1, filterb_2, "NCHW", "VALID", 1, 1)
    covb_3= ad.convolution_2d_forward_op(covb_2, filterb_3, "NCHW", "SAME", 1, 1)
    poolb= ad.pooling_2d_forward_op(covb_3,"NCHW", "max", 0, 0, 2, 2, 3, 3)
    covb_4 = ad.convolution_2d_forward_op(poolb, filterb_4, "NCHW", "VALID", 1, 1)
    covb_5 = ad.convolution_2d_forward_op(covb_4, filterb_5, "NCHW", "VALID", 1, 1)
    covb = ad.pooling_2d_forward_op(covb_5,"NCHW", "max", 0, 0, 2, 2, 3, 3)



# inception_moudle1
 #inception_moudle1_1
    filter1_1_0= ad.Variable("filter1_1_0")
    filter1_1_1a = ad.Variable("filter1_1_1a")
    filter1_1_1b = ad.Variable("filter1_1_1b")
    filter1_1_2a = ad.Variable("filter1_1_2a")
    filter1_1_2b = ad.Variable("filter1_1_2b")
    filter1_1_2c = ad.Variable("filter1_1_2c")
    filter1_1_3 = ad.Variable("filter1_1_3a")

    filter1_1_0_val = np.ones((64, 192, 1, 1)) * 0.01
    filter1_1_1_vala = np.ones((48, 192, 1, 1)) * 0.01
    filter1_1_1_valb = np.ones((64, 48, 5, 5)) * 0.01
    filter1_1_2_vala = np.ones((64, 192, 1, 1)) * 0.01
    filter1_1_2_valb = np.ones((96, 64, 3, 3)) * 0.01
    filter1_1_2_valc = np.ones((96, 96, 3, 3)) * 0.01
    filter1_1_3_val = np.ones((32, 192, 1, 1)) * 0.01

    # branch_0
    incep1_1_0=ad.convolution_2d_forward_op(covb, filter1_1_0, "NCHW", "SAME", 1, 1)
    # branch 1
    incep1_1_1a=ad.convolution_2d_forward_op(covb, filter1_1_1a, "NCHW", "SAME", 1, 1)
    incep1_1_1 = ad.convolution_2d_forward_op(incep1_1_1a, filter1_1_1b, "NCHW", "SAME", 1, 1)
    # branch 2
    incep1_1_2a = ad.convolution_2d_forward_op(covb, filter1_1_2a, "NCHW", "SAME", 1, 1)
    incep1_1_2b = ad.convolution_2d_forward_op(incep1_1_2a, filter1_1_2b, "NCHW", "SAME", 1, 1)
    incep1_1_2 = ad.convolution_2d_forward_op(incep1_1_2b, filter1_1_2c, "NCHW", "SAME", 1, 1)
    # branch 3
    incep1_1_3a = ad.pooling_2d_forward_op(covb,"NCHW", "mean", 1, 1, 1, 1, 3, 3)
    incep1_1_3 = ad.convolution_2d_forward_op(incep1_1_3a, filter1_1_3, "NCHW", "SAME", 1, 1)

    concat1_1a=ad.concat_forward_op(incep1_1_0,incep1_1_1)
    concat1_1b = ad.concat_forward_op(concat1_1a, incep1_1_2)
    concat1_1 = ad.concat_forward_op(concat1_1b, incep1_1_3)

 #inception_moudle1_2
    filter1_2_0 = ad.Variable("filter1_2_0")
    filter1_2_1a = ad.Variable("filter1_2_1a")
    filter1_2_1b = ad.Variable("filter1_2_1b")
    filter1_2_2a = ad.Variable("filter1_2_2a")
    filter1_2_2b = ad.Variable("filter1_2_2b")
    filter1_2_2c = ad.Variable("filter1_2_2c")
    filter1_2_3 = ad.Variable("filter1_2_3a")

    filter1_2_0_val = np.ones((64, 256, 1, 1)) * 0.01
    filter1_2_1_vala = np.ones((48, 256, 1, 1)) * 0.01
    filter1_2_1_valb = np.ones((64, 48, 5, 5)) * 0.01
    filter1_2_2_vala = np.ones((64, 256, 1, 1)) * 0.01
    filter1_2_2_valb = np.ones((96, 64, 3, 3)) * 0.01
    filter1_2_2_valc = np.ones((96, 96, 3, 3)) * 0.01
    filter1_2_3_val = np.ones((64, 256, 1, 1)) * 0.01

    # branch_0
    incep1_2_0 = ad.convolution_2d_forward_op(concat1_1, filter1_2_0, "NCHW", "SAME", 1, 1)
    # branch 1
    incep1_2_1a = ad.convolution_2d_forward_op(concat1_1, filter1_2_1a, "NCHW", "SAME", 1, 1)
    incep1_2_1 = ad.convolution_2d_forward_op(incep1_2_1a, filter1_2_1b, "NCHW", "SAME", 1, 1)
    # branch 2
    incep1_2_2a = ad.convolution_2d_forward_op(concat1_1, filter1_2_2a, "NCHW", "SAME", 1, 1)
    incep1_2_2b = ad.convolution_2d_forward_op(incep1_2_2a, filter1_2_2b, "NCHW", "SAME", 1, 1)
    incep1_2_2 = ad.convolution_2d_forward_op(incep1_2_2b, filter1_2_2c, "NCHW", "SAME", 1, 1)
    # branch 3
    incep1_2_3a = ad.pooling_2d_forward_op(concat1_1, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
    incep1_2_3 = ad.convolution_2d_forward_op(incep1_2_3a, filter1_2_3, "NCHW", "SAME", 1, 1)

    concat1_2a = ad.concat_forward_op(incep1_2_0, incep1_2_1)
    concat1_2b = ad.concat_forward_op(concat1_2a, incep1_2_2)
    concat1_2 = ad.concat_forward_op(concat1_2b, incep1_2_3)

 # inception_moudle1_3
    filter1_3_0 = ad.Variable("filter1_3_0")
    filter1_3_1a = ad.Variable("filter1_3_1a")
    filter1_3_1b = ad.Variable("filter1_3_1b")
    filter1_3_2a = ad.Variable("filter1_3_2a")
    filter1_3_2b = ad.Variable("filter1_3_2b")
    filter1_3_2c = ad.Variable("filter1_3_2c")
    filter1_3_3 = ad.Variable("filter1_3_3a")

    filter1_3_0_val = np.ones((64, 288, 1, 1)) * 0.01
    filter1_3_1_vala = np.ones((48, 288, 1, 1)) * 0.01
    filter1_3_1_valb = np.ones((64, 48, 5, 5)) * 0.01
    filter1_3_2_vala = np.ones((64, 288, 1, 1)) * 0.01
    filter1_3_2_valb = np.ones((96, 64, 3, 3)) * 0.01
    filter1_3_2_valc = np.ones((96, 96, 3, 3)) * 0.01
    filter1_3_3_val = np.ones((64, 288, 1, 1)) * 0.01

    # branch_0
    incep1_3_0 = ad.convolution_2d_forward_op(concat1_2, filter1_3_0, "NCHW", "SAME", 1, 1)
    # branch 1
    incep1_3_1a = ad.convolution_2d_forward_op(concat1_2, filter1_3_1a, "NCHW", "SAME", 1, 1)
    incep1_3_1 = ad.convolution_2d_forward_op(incep1_3_1a, filter1_3_1b, "NCHW", "SAME", 1, 1)
    # branch 2
    incep1_3_2a = ad.convolution_2d_forward_op(concat1_2, filter1_3_2a, "NCHW", "SAME", 1, 1)
    incep1_3_2b = ad.convolution_2d_forward_op(incep1_3_2a, filter1_3_2b, "NCHW", "SAME", 1, 1)
    incep1_3_2 = ad.convolution_2d_forward_op(incep1_3_2b, filter1_3_2c, "NCHW", "SAME", 1, 1)
    # branch 3
    incep1_3_3a = ad.pooling_2d_forward_op(concat1_2, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
    incep1_3_3 = ad.convolution_2d_forward_op(incep1_3_3a, filter1_3_3, "NCHW", "SAME", 1, 1)

    concat1_3a = ad.concat_forward_op(incep1_3_0, incep1_3_1)
    concat1_3b = ad.concat_forward_op(concat1_3a, incep1_3_2)
    concat1_3 = ad.concat_forward_op(concat1_3b, incep1_3_3)

#
#
#
#
# # inception_moudle2
    # inception_moudle2_1
    filter2_1_0 = ad.Variable("filter2_1_0")
    filter2_1_1a = ad.Variable("filter2_1_1a")
    filter2_1_1b = ad.Variable("filter2_1_1b")
    filter2_1_1c = ad.Variable("filter2_1_1c")

    filter2_1_0_val = np.ones((384, 288, 3, 3)) * 0.01
    filter2_1_1_vala = np.ones((64, 288, 1, 1)) * 0.01
    filter2_1_1_valb = np.ones((96, 64, 3, 3)) * 0.01
    filter2_1_1_valc = np.ones((96, 96, 3, 3)) * 0.01

    # branch_0
    incep2_1_0 = ad.convolution_2d_forward_op(concat1_3, filter2_1_0, "NCHW", "VALID", 2, 2)
    # branch 1
    incep2_1_1a = ad.convolution_2d_forward_op(concat1_3, filter2_1_1a, "NCHW", "SAME", 1, 1)
    incep2_1_1b = ad.convolution_2d_forward_op(incep2_1_1a, filter2_1_1b, "NCHW", "SAME", 1, 1)
    incep2_1_1 = ad.convolution_2d_forward_op(incep2_1_1b, filter2_1_1c, "NCHW", "VALID", 2,2)
    # branch 2
    incep2_1_2 = ad.pooling_2d_forward_op(concat1_3, "NCHW", "mean", 0, 0, 2, 2, 3, 3)

    concat2_1a = ad.concat_forward_op(incep2_1_0, incep2_1_1)
    concat2_1 = ad.concat_forward_op(concat2_1a, incep2_1_2)


  # inception_moudle2_2
    filter2_2_0 = ad.Variable("filter2_2_0")
    filter2_2_1a = ad.Variable("filter2_2_1a")
    filter2_2_1b = ad.Variable("filter2_2_1b")
    filter2_2_1c = ad.Variable("filter2_2_1c")
    filter2_2_2a = ad.Variable("filter2_2_2a")
    filter2_2_2b = ad.Variable("filter2_2_2b")
    filter2_2_2c = ad.Variable("filter2_2_2c")
    filter2_2_2d = ad.Variable("filter2_2_2d")
    filter2_2_2e = ad.Variable("filter2_2_2e")
    filter2_2_3 = ad.Variable("filter2_2_3a")

    filter2_2_0_val = np.ones((192, 768, 1, 1)) * 0.01
    filter2_2_1_vala = np.ones((128, 768, 1, 1)) * 0.01
    filter2_2_1_valb = np.ones((128, 128, 1, 7)) * 0.01
    filter2_2_1_valc = np.ones((192, 128, 7, 1)) * 0.01
    filter2_2_2_vala = np.ones((128, 768, 1, 1)) * 0.01
    filter2_2_2_valb = np.ones((128, 128, 7,1)) * 0.01
    filter2_2_2_valc = np.ones((128, 128, 1, 7)) * 0.01
    filter2_2_2_vald = np.ones((128, 128, 7, 1)) * 0.01
    filter2_2_2_vale = np.ones((192, 128, 1, 7)) * 0.01
    filter2_2_3_val = np.ones((192, 768, 1, 1)) * 0.01

    # branch_0
    incep2_2_0 = ad.convolution_2d_forward_op(concat2_1, filter2_2_0, "NCHW", "SAME", 1, 1)
    # branch 1
    incep2_2_1a = ad.convolution_2d_forward_op(concat2_1, filter2_2_1a, "NCHW", "SAME", 1, 1)
    incep2_2_1b = ad.convolution_2d_forward_op(incep2_2_1a, filter2_2_1b, "NCHW", "SAME", 1, 1)
    incep2_2_1 = ad.convolution_2d_forward_op(incep2_2_1b, filter2_2_1c, "NCHW", "SAME", 1, 1)
    # branch 2
    incep2_2_2a = ad.convolution_2d_forward_op(concat2_1, filter2_2_2a, "NCHW", "SAME", 1, 1)
    incep2_2_2b = ad.convolution_2d_forward_op(incep2_2_2a, filter2_2_2b, "NCHW", "SAME", 1, 1)
    incep2_2_2c = ad.convolution_2d_forward_op(incep2_2_2b, filter2_2_2c, "NCHW", "SAME", 1, 1)
    incep2_2_2d = ad.convolution_2d_forward_op(incep2_2_2c, filter2_2_2d, "NCHW", "SAME", 1, 1)
    incep2_2_2 = ad.convolution_2d_forward_op(incep2_2_2d, filter2_2_2e, "NCHW", "SAME", 1, 1)
    # branch 3
    incep2_2_3a = ad.pooling_2d_forward_op(concat2_1, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
    incep2_2_3 = ad.convolution_2d_forward_op(incep2_2_3a, filter2_2_3, "NCHW", "SAME", 1, 1)

    concat2_2a = ad.concat_forward_op(incep2_2_0, incep2_2_1)
    concat2_2b = ad.concat_forward_op(concat2_2a, incep2_2_2)
    concat2_2 = ad.concat_forward_op(concat2_2b, incep2_2_3)

# inception_moudle2_3
#     filter2_3_0 = ad.Variable("filter2_3_0")
#     filter2_3_1a = ad.Variable("filter2_3_1a")
#     filter2_3_1b = ad.Variable("filter2_3_1b")
#     filter2_3_1c = ad.Variable("filter2_3_1c")
#     filter2_3_2a = ad.Variable("filter2_3_2a")
#     filter2_3_2b = ad.Variable("filter2_3_2b")
#     filter2_3_2c = ad.Variable("filter2_3_2c")
#     filter2_3_2d = ad.Variable("filter2_3_2d")
#     filter2_3_2e = ad.Variable("filter2_3_2e")
#     filter2_3_3 = ad.Variable("filter2_3_3a")
#
#     filter2_3_0_val = np.ones((192, 768, 1, 1)) * 0.01
#     filter2_3_1_vala = np.ones((160, 768, 1, 1)) * 0.01
#     filter2_3_1_valb = np.ones((160, 160, 1, 7)) * 0.01
#     filter2_3_1_valc = np.ones((192, 160, 7, 1)) * 0.01
#     filter2_3_2_vala = np.ones((160, 768, 1, 1)) * 0.01
#     filter2_3_2_valb = np.ones((160, 160, 7,1)) * 0.01
#     filter2_3_2_valc = np.ones((160, 160, 1, 7)) * 0.01
#     filter2_3_2_vald = np.ones((160, 160, 7, 1)) * 0.01
#     filter2_3_2_vale = np.ones((192, 160, 1, 7)) * 0.01
#     filter2_3_3_val = np.ones((192, 768, 1, 1)) * 0.01
#
#     # branch_0
#     incep2_3_0 = ad.convolution_2d_forward_op(concat2_2, filter2_3_0, "NCHW", "SAME", 1, 1)
#     # branch 1
#     incep2_3_1a = ad.convolution_2d_forward_op(concat2_2, filter2_3_1a, "NCHW", "SAME", 1, 1)
#     incep2_3_1b = ad.convolution_2d_forward_op(incep2_3_1a, filter2_3_1b, "NCHW", "SAME", 1, 1)
#     incep2_3_1 = ad.convolution_2d_forward_op(incep2_3_1b, filter2_3_1c, "NCHW", "SAME", 1, 1)
#     # branch 2
#     incep2_3_2a = ad.convolution_2d_forward_op(concat2_2, filter2_3_2a, "NCHW", "SAME", 1, 1)
#     incep2_3_2b = ad.convolution_2d_forward_op(incep2_3_2a, filter2_3_2b, "NCHW", "SAME", 1, 1)
#     incep2_3_2c = ad.convolution_2d_forward_op(incep2_3_2b, filter2_3_2c, "NCHW", "SAME", 1, 1)
#     incep2_3_2d = ad.convolution_2d_forward_op(incep2_3_2c, filter2_3_2d, "NCHW", "SAME", 1, 1)
#     incep2_3_2 = ad.convolution_2d_forward_op(incep2_3_2d, filter2_3_2e, "NCHW", "SAME", 1, 1)
#     # branch 3
#     incep2_3_3a = ad.pooling_2d_forward_op(concat2_2, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
#     incep2_3_3 = ad.convolution_2d_forward_op(incep2_3_3a, filter2_3_3, "NCHW", "SAME", 1, 1)
#
#     concat2_3a = ad.concat_forward_op(incep2_3_0, incep2_3_1)
#     concat2_3b = ad.concat_forward_op(concat2_3a, incep2_3_2)
#     concat2_3 = ad.concat_forward_op(concat2_3b, incep2_3_3)


 # inception_moudle2_4
 #    filter2_4_0 = ad.Variable("filter2_4_0")
 #    filter2_4_1a = ad.Variable("filter2_4_1a")
 #    filter2_4_1b = ad.Variable("filter2_4_1b")
 #    filter2_4_1c = ad.Variable("filter2_4_1c")
 #    filter2_4_2a = ad.Variable("filter2_4_2a")
 #    filter2_4_2b = ad.Variable("filter2_4_2b")
 #    filter2_4_2c = ad.Variable("filter2_4_2c")
 #    filter2_4_2d = ad.Variable("filter2_4_2d")
 #    filter2_4_2e = ad.Variable("filter2_4_2e")
 #    filter2_4_3 = ad.Variable("filter2_4_3a")
 #
 #    filter2_4_0_val = np.ones((192, 768, 1, 1)) * 0.01
 #    filter2_4_1_vala = np.ones((160, 768, 1, 1)) * 0.01
 #    filter2_4_1_valb = np.ones((160, 160, 1, 7)) * 0.01
 #    filter2_4_1_valc = np.ones((192, 160, 7, 1)) * 0.01
 #    filter2_4_2_vala = np.ones((160, 768, 1, 1)) * 0.01
 #    filter2_4_2_valb = np.ones((160, 160, 7, 1)) * 0.01
 #    filter2_4_2_valc = np.ones((160, 160, 1, 7)) * 0.01
 #    filter2_4_2_vald = np.ones((160, 160, 7, 1)) * 0.01
 #    filter2_4_2_vale = np.ones((192, 160, 1, 7)) * 0.01
 #    filter2_4_3_val = np.ones((192, 768, 1, 1)) * 0.01
 #
 #    # branch_0
 #    incep2_4_0 = ad.convolution_2d_forward_op(concat2_3, filter2_4_0, "NCHW", "SAME", 1, 1)
 #    # branch 1
 #    incep2_4_1a = ad.convolution_2d_forward_op(concat2_3, filter2_4_1a, "NCHW", "SAME", 1, 1)
 #    incep2_4_1b = ad.convolution_2d_forward_op(incep2_4_1a, filter2_4_1b, "NCHW", "SAME", 1, 1)
 #    incep2_4_1 = ad.convolution_2d_forward_op(incep2_4_1b, filter2_4_1c, "NCHW", "SAME", 1, 1)
 #    # branch 2
 #    incep2_4_2a = ad.convolution_2d_forward_op(concat2_3, filter2_4_2a, "NCHW", "SAME", 1, 1)
 #    incep2_4_2b = ad.convolution_2d_forward_op(incep2_4_2a, filter2_4_2b, "NCHW", "SAME", 1, 1)
 #    incep2_4_2c = ad.convolution_2d_forward_op(incep2_4_2b, filter2_4_2c, "NCHW", "SAME", 1, 1)
 #    incep2_4_2d = ad.convolution_2d_forward_op(incep2_4_2c, filter2_4_2d, "NCHW", "SAME", 1, 1)
 #    incep2_4_2 = ad.convolution_2d_forward_op(incep2_4_2d, filter2_4_2e, "NCHW", "SAME", 1, 1)
 #    # branch 3
 #    incep2_4_3a = ad.pooling_2d_forward_op(concat2_3, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
 #    incep2_4_3 = ad.convolution_2d_forward_op(incep2_4_3a, filter2_4_3, "NCHW", "SAME", 1, 1)
 #
 #    concat2_4a = ad.concat_forward_op(incep2_4_0, incep2_4_1)
 #    concat2_4b = ad.concat_forward_op(concat2_4a, incep2_4_2)
 #    concat2_4 = ad.concat_forward_op(concat2_4b, incep2_4_3)


# inception_moudle2_5
#     filter2_5_0 = ad.Variable("filter2_5_0")
#     filter2_5_1a = ad.Variable("filter2_5_1a")
#     filter2_5_1b = ad.Variable("filter2_5_1b")
#     filter2_5_1c = ad.Variable("filter2_5_1c")
#     filter2_5_2a = ad.Variable("filter2_5_2a")
#     filter2_5_2b = ad.Variable("filter2_5_2b")
#     filter2_5_2c = ad.Variable("filter2_5_2c")
#     filter2_5_2d = ad.Variable("filter2_5_2d")
#     filter2_5_2e = ad.Variable("filter2_5_2e")
#     filter2_5_3 = ad.Variable("filter2_5_3a")
#
#     filter2_5_0_val = np.ones((192, 768, 1, 1)) * 0.01
#     filter2_5_1_vala = np.ones((160, 768, 1, 1)) * 0.01
#     filter2_5_1_valb = np.ones((160, 160, 1, 7)) * 0.01
#     filter2_5_1_valc = np.ones((192, 160, 7, 1)) * 0.01
#     filter2_5_2_vala = np.ones((160, 768, 1, 1)) * 0.01
#     filter2_5_2_valb = np.ones((160, 160, 7, 1)) * 0.01
#     filter2_5_2_valc = np.ones((160, 160, 1, 7)) * 0.01
#     filter2_5_2_vald = np.ones((160, 160, 7, 1)) * 0.01
#     filter2_5_2_vale = np.ones((192, 160, 1, 7)) * 0.01
#     filter2_5_3_val = np.ones((192, 768, 1, 1)) * 0.01
#
#     # branch_0
#     incep2_5_0 = ad.convolution_2d_forward_op(concat2_4, filter2_5_0, "NCHW", "SAME", 1, 1)
#     # branch 1
#     incep2_5_1a = ad.convolution_2d_forward_op(concat2_4, filter2_5_1a, "NCHW", "SAME", 1, 1)
#     incep2_5_1b = ad.convolution_2d_forward_op(incep2_5_1a, filter2_5_1b, "NCHW", "SAME", 1, 1)
#     incep2_5_1 = ad.convolution_2d_forward_op(incep2_5_1b, filter2_5_1c, "NCHW", "SAME", 1, 1)
#     # branch 2
#     incep2_5_2a = ad.convolution_2d_forward_op(concat2_4, filter2_5_2a, "NCHW", "SAME", 1, 1)
#     incep2_5_2b = ad.convolution_2d_forward_op(incep2_5_2a, filter2_5_2b, "NCHW", "SAME", 1, 1)
#     incep2_5_2c = ad.convolution_2d_forward_op(incep2_5_2b, filter2_5_2c, "NCHW", "SAME", 1, 1)
#     incep2_5_2d = ad.convolution_2d_forward_op(incep2_5_2c, filter2_5_2d, "NCHW", "SAME", 1, 1)
#     incep2_5_2 = ad.convolution_2d_forward_op(incep2_5_2d, filter2_5_2e, "NCHW", "SAME", 1, 1)
#     # branch 3
#     incep2_5_3a = ad.pooling_2d_forward_op(concat2_4, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
#     incep2_5_3 = ad.convolution_2d_forward_op(incep2_5_3a, filter2_5_3, "NCHW", "SAME", 1, 1)
#
#     concat2_5a = ad.concat_forward_op(incep2_5_0, incep2_5_1)
#     concat2_5b = ad.concat_forward_op(concat2_5a, incep2_5_2)
#     concat2_5 = ad.concat_forward_op(concat2_5b, incep2_5_3)





# # inception_moudle3
    # inception_moudle3_1
    filter3_1_0a = ad.Variable("filter3_1_0a")
    filter3_1_0b = ad.Variable("filter3_1_0b")
    filter3_1_1a = ad.Variable("filter3_1_1a")
    filter3_1_1b = ad.Variable("filter3_1_1b")
    filter3_1_1c = ad.Variable("filter3_1_1c")
    filter3_1_1d = ad.Variable("filter3_1_1d")

    filter3_1_0_vala = np.ones((192, 768, 1, 1)) * 0.01
    filter3_1_0_valb = np.ones((320, 192, 3, 3)) * 0.01
    filter3_1_1_vala = np.ones((192, 768, 1, 1)) * 0.01
    filter3_1_1_valb = np.ones((192, 192, 1,7)) * 0.01
    filter3_1_1_valc = np.ones((192, 192, 7, 1)) * 0.01
    filter3_1_1_vald = np.ones((192, 192, 3, 3)) * 0.01

    # branch_0
    incep3_1_0a = ad.convolution_2d_forward_op(concat2_2, filter3_1_0a, "NCHW", "SAME", 1, 1)
    incep3_1_0 = ad.convolution_2d_forward_op(incep3_1_0a, filter3_1_0b, "NCHW", "VALID", 2,2)
    # branch 1
    incep3_1_1a = ad.convolution_2d_forward_op(concat2_2, filter3_1_1a, "NCHW", "SAME", 1, 1)
    incep3_1_1b = ad.convolution_2d_forward_op(incep3_1_1a, filter3_1_1b, "NCHW", "SAME", 1, 1)
    incep3_1_1c = ad.convolution_2d_forward_op(incep3_1_1b, filter3_1_1c, "NCHW", "SAME", 1, 1)
    incep3_1_1 = ad.convolution_2d_forward_op(incep3_1_1c, filter3_1_1d, "NCHW", "VALID", 2, 2)
    # branch 2
    incep3_1_2 = ad.pooling_2d_forward_op(concat2_2, "NCHW", "mean", 0, 0, 2, 2, 3, 3)

    concat3_1a = ad.concat_forward_op(incep3_1_0, incep3_1_1)
    concat3_1 = ad.concat_forward_op(concat3_1a, incep3_1_2)



# inception_moudle3_2
    filter3_2_0 = ad.Variable("filter3_2_0")
    filter3_2_1a = ad.Variable("filter3_2_1a")
    filter3_2_1b = ad.Variable("filter3_2_1b")
    filter3_2_1c = ad.Variable("filter3_2_1c")
    filter3_2_2a = ad.Variable("filter3_2_2a")
    filter3_2_2b = ad.Variable("filter3_2_2b")
    filter3_2_2c = ad.Variable("filter3_2_2c")
    filter3_2_2d = ad.Variable("filter3_2_2d")
    filter3_2_3 = ad.Variable("filter3_2_3a")

    filter3_2_0_val = np.ones((320, 1280, 1, 1)) * 0.01
    filter3_2_1_vala = np.ones((384, 1280, 1, 1)) * 0.01
    filter3_2_1_valb = np.ones((384, 384, 1, 3)) * 0.01
    filter3_2_1_valc = np.ones((384, 384, 3, 1)) * 0.01
    filter3_2_2_vala = np.ones((448, 1280, 1, 1)) * 0.01
    filter3_2_2_valb = np.ones((384, 448, 3, 3)) * 0.01
    filter3_2_2_valc = np.ones((384, 384, 1, 3)) * 0.01
    filter3_2_2_vald = np.ones((384, 384, 3, 1)) * 0.01
    filter3_2_3_val = np.ones((192, 1280, 1, 1)) * 0.01

    # branch_0
    incep3_2_0 = ad.convolution_2d_forward_op(concat3_1, filter3_2_0, "NCHW", "SAME", 1, 1)
    # branch 1
    incep3_2_1a = ad.convolution_2d_forward_op(concat3_1, filter3_2_1a, "NCHW", "SAME", 1, 1)
    incep3_2_1b = ad.convolution_2d_forward_op(incep3_2_1a, filter3_2_1b, "NCHW", "SAME", 1, 1)
    incep3_2_1c = ad.convolution_2d_forward_op(incep3_2_1a, filter3_2_1c, "NCHW", "SAME", 1, 1)
    incep3_2_1=ad.concat_forward_op(incep3_2_1b, incep3_2_1c)
    # branch 2
    incep3_2_2a = ad.convolution_2d_forward_op(concat3_1, filter3_2_2a, "NCHW", "SAME", 1, 1)
    incep3_2_2b = ad.convolution_2d_forward_op(incep3_2_2a, filter3_2_2b, "NCHW", "SAME", 1, 1)
    incep3_2_2c = ad.convolution_2d_forward_op(incep3_2_2b, filter3_2_2c, "NCHW", "SAME", 1, 1)
    incep3_2_2d = ad.convolution_2d_forward_op(incep3_2_2b, filter3_2_2d, "NCHW", "SAME", 1, 1)
    incep3_2_2 = ad.concat_forward_op(incep3_2_2c, incep3_2_2d)
    # branch 3
    incep3_2_3a = ad.pooling_2d_forward_op(concat3_1, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
    incep3_2_3 = ad.convolution_2d_forward_op(incep3_2_3a, filter3_2_3, "NCHW", "SAME", 1, 1)

    concat3_2a = ad.concat_forward_op(incep3_2_0, incep3_2_1)
    concat3_2b = ad.concat_forward_op(concat3_2a, incep3_2_2)
    concat3_2 = ad.concat_forward_op(concat3_2b, incep3_2_3)


# # inception_moudle3_3
#     filter3_3_0 = ad.Variable("filter3_3_0")
#     filter3_3_1a = ad.Variable("filter3_3_1a")
#     filter3_3_1b = ad.Variable("filter3_3_1b")
#     filter3_3_1c = ad.Variable("filter3_3_1c")
#     filter3_3_2a = ad.Variable("filter3_3_2a")
#     filter3_3_2b = ad.Variable("filter3_3_2b")
#     filter3_3_2c = ad.Variable("filter3_3_2c")
#     filter3_3_2d = ad.Variable("filter3_3_2d")
#     filter3_3_3 = ad.Variable("filter3_3_3a")
#
#     filter3_3_0_val = np.ones((320, 2048, 1, 1)) * 0.01
#     filter3_3_1_vala = np.ones((384, 2048, 1, 1)) * 0.01
#     filter3_3_1_valb = np.ones((384, 384, 1, 3)) * 0.01
#     filter3_3_1_valc = np.ones((384, 384, 3, 1)) * 0.01
#     filter3_3_2_vala = np.ones((448, 2048, 1, 1)) * 0.01
#     filter3_3_2_valb = np.ones((384, 448, 3, 3)) * 0.01
#     filter3_3_2_valc = np.ones((384, 384, 1, 3)) * 0.01
#     filter3_3_2_vald = np.ones((384, 384, 3, 1)) * 0.01
#     filter3_3_3_val = np.ones((192, 2048, 1, 1)) * 0.01
#
#     # branch_0
#     incep3_3_0 = ad.convolution_2d_forward_op(concat3_2, filter3_3_0, "NCHW", "SAME", 1, 1)
#     # branch 1
#     incep3_3_1a = ad.convolution_2d_forward_op(concat3_2, filter3_3_1a, "NCHW", "SAME", 1, 1)
#     incep3_3_1b = ad.convolution_2d_forward_op(incep3_3_1a, filter3_3_1b, "NCHW", "SAME", 1, 1)
#     incep3_3_1c = ad.convolution_2d_forward_op(incep3_3_1a, filter3_3_1c, "NCHW", "SAME", 1, 1)
#     incep3_3_1 = ad.concat_forward_op(incep3_3_1b, incep3_3_1c)
#
#     # branch 2
#     incep3_3_2a = ad.convolution_2d_forward_op(concat3_2, filter3_3_2a, "NCHW", "SAME", 1, 1)
#     incep3_3_2b = ad.convolution_2d_forward_op(incep3_3_2a, filter3_3_2b, "NCHW", "SAME", 1, 1)
#     incep3_3_2c = ad.convolution_2d_forward_op(incep3_3_2b, filter3_3_2c, "NCHW", "SAME", 1, 1)
#     incep3_3_2d = ad.convolution_2d_forward_op(incep3_3_2b, filter3_3_2d, "NCHW", "SAME", 1, 1)
#     incep3_3_2 = ad.concat_forward_op(incep3_3_2c, incep3_3_2d)
#     # branch 3
#     incep3_3_3a = ad.pooling_2d_forward_op(concat3_2, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
#     incep3_3_3 = ad.convolution_2d_forward_op(incep3_3_3a, filter3_3_3, "NCHW", "SAME", 1, 1)
#
#     concat3_3a = ad.concat_forward_op(incep3_3_0, incep3_3_1)
#     concat3_3b = ad.concat_forward_op(concat3_3a, incep3_3_2)
#     concat3_3 = ad.concat_forward_op(concat3_3b, incep3_3_3)

    filtera1 = ad.Variable("filtera1")
    filtera1val = np.ones((1000, 2048, 1, 1)) * 0.01

    filtersmul = ad.Variable("filtersmul")
    filtersmulval = np.ones((1000, n_class)) * 0.01

    biases= ad.Variable("biases")
    biasesval = np.ones((batch_size, n_class)) * 0.01


    poollast=ad.pooling_2d_forward_op(concat3_2,"NCHW", "mean",0,0,1,1,8,8)
    dropout=ad.dropout_forward_op(poollast,"NCHW",0.8)
    convlast=ad.convolution_2d_forward_op(dropout, filtera1, "NCHW", "SAME", 1, 1)
    squeeze=ad.squeeze_op(convlast)

    # fc8
    mul8 = ad.matmul_op(squeeze, filtersmul)
    add8 = ad.add_op(mul8, biases)
    fc8 = ad.fullyactivation_forward_op(add8, "NCHW", "softmax")

    loss = ad.softmaxcrossentropy_op(fc8, y_)
    X_val = np.empty(shape=(batch_size, 3, 32, 32), dtype=np.float32)
    y_val = np.empty(shape=(batch_size, n_class), dtype=np.float32)
    aph = 0.001
    t = train.Adam_minimize(loss, aph)
    t.init_Variable(
        { filterb_1: filtersb_val1, filterb_2: filtersb_val2, filterb_3: filtersb_val3
                                       , filterb_4: filtersb_val4, filterb_5: filtersb_val5,
         filter1_1_0:filter1_1_0_val ,filter1_1_1a:filter1_1_1_vala,filter1_1_1b:filter1_1_1_valb,filter1_1_2a:filter1_1_2_vala,filter1_1_2b:filter1_1_2_valb
                                       ,filter1_1_2c:filter1_1_2_valc,filter1_1_3:filter1_1_3_val
         ,filter1_2_0: filter1_2_0_val, filter1_2_1a: filter1_2_1_vala,
                                       filter1_2_1b: filter1_2_1_valb, filter1_2_2a: filter1_2_2_vala,
                                       filter1_2_2b: filter1_2_2_valb, filter1_2_2c: filter1_2_2_valc, filter1_2_3: filter1_2_3_val

        , filter1_3_0: filter1_3_0_val, filter1_3_1a: filter1_3_1_vala,
                                       filter1_3_1b: filter1_3_1_valb, filter1_3_2a: filter1_3_2_vala,
                                       filter1_3_2b: filter1_3_2_valb, filter1_3_2c: filter1_3_2_valc,
                                       filter1_3_3: filter1_3_3_val
        ,filter2_1_0:filter2_1_0_val,filter2_1_1a:filter2_1_1_vala,filter2_1_1b:filter2_1_1_valb,filter2_1_1c:filter2_1_1_valc

        ,filter2_2_0:filter2_2_0_val,filter2_2_1a:filter2_2_1_vala,filter2_2_1b:filter2_2_1_valb,filter2_2_1c:filter2_2_1_valc,
                                       filter2_2_2a:filter2_2_2_vala,filter2_2_2b:filter2_2_2_valb,filter2_2_2c:filter2_2_2_valc,filter2_2_2d:filter2_2_2_vald,filter2_2_2e:filter2_2_2_vale,filter2_2_3:filter2_2_3_val

        # , filter2_3_0: filter2_3_0_val, filter2_3_1a: filter2_3_1_vala, filter2_3_1b: filter2_3_1_valb,
        #                                filter2_3_1c: filter2_3_1_valc,
        #                                filter2_3_2a: filter2_3_2_vala, filter2_3_2b: filter2_3_2_valb,
        #                                filter2_3_2c: filter2_3_2_valc, filter2_3_2d: filter2_3_2_vald,
        #                                filter2_3_2e: filter2_3_2_vale, filter2_3_3: filter2_3_3_val
        # , filter2_4_0: filter2_4_0_val, filter2_4_1a: filter2_4_1_vala, filter2_4_1b: filter2_4_1_valb,
        #                                filter2_4_1c: filter2_4_1_valc,
        #                                filter2_4_2a: filter2_4_2_vala, filter2_4_2b: filter2_4_2_valb,
        #                                filter2_4_2c: filter2_4_2_valc, filter2_4_2d: filter2_4_2_vald,
        #                                filter2_4_2e: filter2_4_2_vale, filter2_4_3: filter2_4_3_val
        # , filter2_5_0: filter2_5_0_val, filter2_5_1a: filter2_5_1_vala, filter2_5_1b: filter2_5_1_valb,
        #                                filter2_5_1c: filter2_5_1_valc,
        #                                filter2_5_2a: filter2_5_2_vala, filter2_5_2b: filter2_5_2_valb,
        #                                filter2_5_2c: filter2_5_2_valc, filter2_5_2d: filter2_5_2_vald,
        #                                filter2_5_2e: filter2_5_2_vale, filter2_5_3: filter2_5_3_val
        , filter3_1_0a: filter3_1_0_vala,filter3_1_0b: filter3_1_0_valb,  filter3_1_1a: filter3_1_1_vala, filter3_1_1b: filter3_1_1_valb,
                                       filter3_1_1c: filter3_1_1_valc,filter3_1_1d: filter3_1_1_vald
        , filter3_2_0: filter3_2_0_val, filter3_2_1a: filter3_2_1_vala,
                                       filter3_2_1b: filter3_2_1_valb,
                                       filter3_2_1c: filter3_2_1_valc,filter3_2_2a: filter3_2_2_vala,filter3_2_2b: filter3_2_2_valb,
                                       filter3_2_2c: filter3_2_2_valc,filter3_2_2d: filter3_2_2_vald,filter3_2_3: filter3_2_3_val
        # , filter3_3_0: filter3_3_0_val, filter3_3_1a: filter3_3_1_vala,
        #                                filter3_3_1b: filter3_3_1_valb,
        #                                filter3_3_1c: filter3_3_1_valc, filter3_3_2a: filter3_3_2_vala,
        #                                filter3_3_2b: filter3_3_2_valb,
        #                                filter3_3_2c: filter3_3_2_valc, filter3_3_2d: filter3_3_2_vald,
        #                                filter3_3_3: filter3_3_3_val,
        ,filtera1:filtera1val,filtersmul:filtersmulval,biases:biasesval})

    train_x, train_y, test_x, test_y = prepare_data()
    num_epochs = 10
    n_train_batches = train_x.shape[0] // batch_size
    n_test_batches = test_x.shape[0] // batch_size
    for i in range(num_epochs):
        print("epoch %d" % i)
        for minibatch_index in range(n_train_batches):
            minibatch_start = minibatch_index * batch_size
            minibatch_end = (minibatch_index + 1) * batch_size
            X_val[:] = train_x[minibatch_start:minibatch_end]
            y_val[:] = train_y[minibatch_start:minibatch_end]
            # print(y_val.shape)
            # y_val[:] = convert_to_one_hot(train_y[minibatch_start:minibatch_end])
            t.run({X: X_val, y_: y_val})
            # print(t.run_get_nodelist_once({X: X_val, y_: y_val}, [fc8])[fc8].asnumpy())
            # print(map_to_numpy(t.get_Variable_node_to_val_map()))
            print("loss_val:", t.get_loss().asnumpy())




    # executor = ad.Executor([answer], ctx=ctx)
    # loss_val = executor.run(feed_dict={inputs: x_val, filterb_1: filtersb_val1, filterb_2: filtersb_val2, filterb_3: filtersb_val3
    #                                    , filterb_4: filtersb_val4, filterb_5: filtersb_val5,
    #      filter1_1_0:filter1_1_0_val ,filter1_1_1a:filter1_1_1_vala,filter1_1_1b:filter1_1_1_valb,filter1_1_2a:filter1_1_2_vala,filter1_1_2b:filter1_1_2_valb
    #                                    ,filter1_1_2c:filter1_1_2_valc,filter1_1_3:filter1_1_3_val
    #      ,filter1_2_0: filter1_2_0_val, filter1_2_1a: filter1_2_1_vala,
    #                                    filter1_2_1b: filter1_2_1_valb, filter1_2_2a: filter1_2_2_vala,
    #                                    filter1_2_2b: filter1_2_2_valb, filter1_2_2c: filter1_2_2_valc, filter1_2_3: filter1_2_3_val
    #
    #     , filter1_3_0: filter1_3_0_val, filter1_3_1a: filter1_3_1_vala,
    #                                    filter1_3_1b: filter1_3_1_valb, filter1_3_2a: filter1_3_2_vala,
    #                                    filter1_3_2b: filter1_3_2_valb, filter1_3_2c: filter1_3_2_valc,
    #                                    filter1_3_3: filter1_3_3_val
    #     ,filter2_1_0:filter2_1_0_val,filter2_1_1a:filter2_1_1_vala,filter2_1_1b:filter2_1_1_valb,filter2_1_1c:filter2_1_1_valc
    #
    #     ,filter2_2_0:filter2_2_0_val,filter2_2_1a:filter2_2_1_vala,filter2_2_1b:filter2_2_1_valb,filter2_2_1c:filter2_2_1_valc,
    #                                    filter2_2_2a:filter2_2_2_vala,filter2_2_2b:filter2_2_2_valb,filter2_2_2c:filter2_2_2_valc,filter2_2_2d:filter2_2_2_vald,filter2_2_2e:filter2_2_2_vale,filter2_2_3:filter2_2_3_val
    #
    #     , filter2_3_0: filter2_3_0_val, filter2_3_1a: filter2_3_1_vala, filter2_3_1b: filter2_3_1_valb,
    #                                    filter2_3_1c: filter2_3_1_valc,
    #                                    filter2_3_2a: filter2_3_2_vala, filter2_3_2b: filter2_3_2_valb,
    #                                    filter2_3_2c: filter2_3_2_valc, filter2_3_2d: filter2_3_2_vald,
    #                                    filter2_3_2e: filter2_3_2_vale, filter2_3_3: filter2_3_3_val
    #     # , filter2_4_0: filter2_4_0_val, filter2_4_1a: filter2_4_1_vala, filter2_4_1b: filter2_4_1_valb,
    #     #                                filter2_4_1c: filter2_4_1_valc,
    #     #                                filter2_4_2a: filter2_4_2_vala, filter2_4_2b: filter2_4_2_valb,
    #     #                                filter2_4_2c: filter2_4_2_valc, filter2_4_2d: filter2_4_2_vald,
    #     #                                filter2_4_2e: filter2_4_2_vale, filter2_4_3: filter2_4_3_val
    #     # , filter2_5_0: filter2_5_0_val, filter2_5_1a: filter2_5_1_vala, filter2_5_1b: filter2_5_1_valb,
    #     #                                filter2_5_1c: filter2_5_1_valc,
    #     #                                filter2_5_2a: filter2_5_2_vala, filter2_5_2b: filter2_5_2_valb,
    #     #                                filter2_5_2c: filter2_5_2_valc, filter2_5_2d: filter2_5_2_vald,
    #     #                                filter2_5_2e: filter2_5_2_vale, filter2_5_3: filter2_5_3_val
    #     , filter3_1_0a: filter3_1_0_vala,filter3_1_0b: filter3_1_0_valb,  filter3_1_1a: filter3_1_1_vala, filter3_1_1b: filter3_1_1_valb,
    #                                    filter3_1_1c: filter3_1_1_valc,filter3_1_1d: filter3_1_1_vald
    #     , filter3_2_0: filter3_2_0_val, filter3_2_1a: filter3_2_1_vala,
    #                                    filter3_2_1b: filter3_2_1_valb,
    #                                    filter3_2_1c: filter3_2_1_valc,filter3_2_2a: filter3_2_2_vala,filter3_2_2b: filter3_2_2_valb,
    #                                    filter3_2_2c: filter3_2_2_valc,filter3_2_2d: filter3_2_2_vald,filter3_2_3: filter3_2_3_val
    #     , filter3_3_0: filter3_3_0_val, filter3_3_1a: filter3_3_1_vala,
    #                                    filter3_3_1b: filter3_3_1_valb,
    #                                    filter3_3_1c: filter3_3_1_valc, filter3_3_2a: filter3_3_2_vala,
    #                                    filter3_3_2b: filter3_3_2_valb,
    #                                    filter3_3_2c: filter3_3_2_valc, filter3_3_2d: filter3_3_2_vald,
    #                                    filter3_3_3: filter3_3_3_val,
    # #     filtera1:filtera1val
    #                                 })

inception_v3()