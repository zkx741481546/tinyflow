import gzip
import os
import time

import numpy as np
import six.moves.cPickle as pickle

from pycode.tinyflow import autodiff as ad
from pycode.tinyflow import gpu_op
from pycode.tinyflow import ndarray
from pycode.tinyflow import TrainExecute

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# np.set_printoptions(threshold=np.inf)

def load_mnist_data(dataset):
    """ Load the dataset
    Code adapted from http://deeplearning.net/tutorial/code/logistic_sgd.py

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
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
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), np.float32
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector), np.int64 that has the same length
    # as the number of rows in the input. It should give the target
    # to the example with the same index in the input.
    return train_set, valid_set, test_set

def convert_to_one_hot(vals):
    """Helper method to convert label array to one-hot array."""
    one_hot_vals = np.zeros((vals.size, vals.max()+1))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals

def sgd_update_gpu(param, grad_param, learning_rate):
    """Helper GPU SGD update method. Avoids copying NDArray to cpu."""
    assert isinstance(param, ndarray.NDArray)
    assert isinstance(grad_param, ndarray.NDArray)
    gpu_op.matrix_elementwise_multiply_by_const(
        grad_param, -learning_rate, grad_param)
    gpu_op.matrix_elementwise_add(param, grad_param, param)

def LeNet5(num_step = 10, print_loss_val_each_epoch = False):

    W1 = ad.Variable("W1")
    W2 = ad.Variable("W2")
    W3 = ad.Variable("W3")
    W4 = ad.Variable("W4")
    b1 = ad.Variable("b1")
    b2 = ad.Variable("b2")
    b3 = ad.Variable("b3")
    b4 = ad.Variable("b4")
    X = ad.Placeholder("X")
    y_ = ad.Placeholder("y_")

    # conv1
    conv1 = ad.conv2withbias(X, W1, b1, "NCHW", "SAME", 1, 1)
    bn1 = ad.bn_forward_op(conv1, "NCHW", "pre_activation")
    pool1 = ad.pooling_2d_forward_op(bn1, "NCHW", "max", 0, 0, 2, 2, 2, 2)

    # conv2
    conv2 = ad.conv2withbias(pool1, W2, b2, "NCHW", "SAME", 1, 1)
    bn2 = ad.bn_forward_op(conv2, "NCHW", "pre_activation")
    pool2 = ad.pooling_2d_forward_op(bn2, "NCHW", "max", 0, 0, 2, 2, 2, 2)

    # fc3
    pool2_flat = ad.flatten_op(pool2)
    fc3 = ad.dense(pool2_flat, W3, b3)
    bn3 = ad.fullybn_forward_op(fc3, "NCHW")
    act3 = ad.fullyactivation_forward_op(bn3, "NCHW", "relu")

    #fc4
    fc4 = ad.dense(act3, W4, b4)
    bn4 = ad.fullybn_forward_op(fc4, "NCHW")
    act4 = ad.fullyactivation_forward_op(bn4, "NCHW", "softmax")
    loss = ad.crossEntropy_loss(act4, y_, False)



    batch_size = 100
    X_val = np.empty(shape=(batch_size, 1, 28, 28), dtype=np.float32)
    y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)
    valid_X_val = np.empty(shape=(batch_size, 1, 28, 28), dtype=np.float32)
    valid_y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)

    rand = np.random.RandomState(seed=123)
    W1_val = rand.normal(scale=0.1, size=(32, 1, 5, 5))
    W2_val = rand.normal(scale=0.1, size=(64, 32, 5, 5))
    W3_val = rand.normal(scale=0.1, size=(7*7*64, 1024))
    W4_val = rand.normal(scale=0.1, size=(1024, 10))
    b1_val = np.ones(32)*0.1
    b2_val = np.ones(64)*0.1
    b3_val = np.ones(1024)*0.1
    b4_val = np.ones(10)*0.1

    # executor_ctx_cpu = ndarray.cpu(0)
    # W1_val = ndarray.array(W1_val, ctx=executor_ctx_cpu)
    # W2_val = ndarray.array(W2_val, ctx=executor_ctx_cpu)
    # W3_val = ndarray.array(W3_val, ctx=executor_ctx_cpu)
    # W4_val = ndarray.array(W4_val, ctx=executor_ctx_cpu)
    # b1_val = ndarray.array(b1_val, ctx=executor_ctx_cpu)
    # b2_val = ndarray.array(b2_val, ctx=executor_ctx_cpu)
    # b3_val = ndarray.array(b3_val, ctx=executor_ctx_cpu)
    # b4_val = ndarray.array(b4_val, ctx=executor_ctx_cpu)
    # X_val = ndarray.array(X_val, ctx=executor_ctx_cpu)
    # y_val = ndarray.array(y_val, ctx=executor_ctx_cpu)




    datasets = load_mnist_data("mnist.pkl.gz")
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    print("train_size", train_set_x.shape[0])
    print("valid_size", valid_set_x.shape[0])
    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size
    train_set_x = train_set_x.reshape(train_set_x.shape[0], 1, 28, 28)
    valid_set_x = valid_set_x.reshape(valid_set_x.shape[0], 1, 28, 28)

    isrun = 0

    aph = 1
    t = TrainExecute.TrainExecutor(loss, aph, ctx = ndarray.gpu(0))
    t.init_Variable({W1: W1_val,
                     W2: W2_val,
                     W3: W3_val,
                     W4: W4_val,
                     b1: b1_val,
                     b2: b2_val,
                     b3: b3_val,
                     b4: b4_val})
    print("==========================================================")

    tic = time.time()
    i = 0
    while i < num_step:

        for minibatch_index in range(n_train_batches):
            minibatch_start = minibatch_index * batch_size
            minibatch_end = (minibatch_index + 1) * batch_size
            X_val[:] = train_set_x[minibatch_start:minibatch_end]
            y_val[:] = convert_to_one_hot(train_set_y[minibatch_start:minibatch_end])

            t.run({X: X_val, y_: y_val})


            if i % 100 == 0:
            #     # print(t.get_loss().asnumpy())
            #
                print("step %d" % i)
                toc = time.time()
                print("use time: " + str(toc - tic))

                correct_predictions = []
                for minibatch_index1 in range(n_valid_batches):
                    minibatch_start1 = minibatch_index1 * batch_size
                    minibatch_end1 = (minibatch_index1 + 1) * batch_size

                    valid_X_val[:] = valid_set_x[minibatch_start1:minibatch_end1]

                    # print("start, end", minibatch_start1, minibatch_end1)
                    # print(convert_to_one_hot(valid_set_y[minibatch_start1:minibatch_end1]).shape)
                    valid_y_val[:] = convert_to_one_hot(valid_set_y[minibatch_start1:minibatch_end1])

                    feed_dict = {X: valid_X_val, y_: valid_y_val}

                    valid_y_predicted = t.run(feed_dict, act4)
                    valid_y_predicted = valid_y_predicted[len(valid_y_predicted)-1].asnumpy()



                    correct_prediction = np.equal(
                        np.argmax(valid_y_val, 1),
                        np.argmax(valid_y_predicted, 1)).astype(np.float)
                    correct_predictions.extend(correct_prediction)
                accuracy = np.mean(correct_predictions)
                print("validation set accuracy=%f" % accuracy)

                tic = time.time()
            #
            #     if(i == 2000):
            #         time.sleep(20)

            i = i + 1
            if i >= num_step:
                break






LeNet5(10000, True)

