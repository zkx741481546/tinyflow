from pycode.tinyflow import autodiff as ad
import numpy as np
from pycode.tinyflow import ndarray
import random
from pycode.tinyflow import gpu_op

def sgd_update_gpu(param, grad_param, learning_rate):
    """Helper GPU SGD update method. Avoids copying NDArray to cpu."""
    assert isinstance(param, ndarray.NDArray)
    assert isinstance(grad_param, ndarray.NDArray)
    gpu_op.matrix_elementwise_multiply_by_const(
        grad_param, -learning_rate, grad_param)
    gpu_op.matrix_elementwise_add(param, grad_param, param)


def vgg16():

    n = 10
    n_class = 10

    inputs = ad.Variable("inputs")
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
    biases6 = ad.Variable("biases6")
    biases7 = ad.Variable("biases7")
    biases8 = ad.Variable("biases8")
    y_ = ad.Variable(name="y_")

    x_val = np.linspace(0, 0.001, 10*3*224*224).reshape((10, 3, 224, 224))
    filters_val = [np.ones((64, 3, 3, 3))*0.001]
    filters_val.append(np.ones((64, 64, 3, 3))*0.001)
    filters_val.append(np.ones((128, 64, 3, 3))*0.001)
    filters_val.append(np.ones((128, 128, 3, 3))*0.001)
    filters_val.append(np.ones((256, 128, 3, 3))*0.001)
    filters_val.append(np.ones((256, 256, 3, 3))*0.001)
    filters_val.append(np.ones((256, 256, 3, 3))*0.001)
    filters_val.append(np.ones((512, 256, 3, 3))*0.001)
    filters_val.append(np.ones((512, 512, 3, 3))*0.001)
    filters_val.append(np.ones((512, 512, 3, 3))*0.001)
    filters_val.append(np.ones((512, 512, 3, 3))*0.001)
    filters_val.append(np.ones((512, 512, 3, 3))*0.001)
    filters_val.append(np.ones((512, 512, 3, 3))*0.001)
    filters_val.append(np.ones((512*7*7, 4096)) * 0.001)
    filters_val.append(np.ones((4096, 4096)) * 0.001)
    filters_val.append(np.ones((4096, n_class)) * 0.001)
    biases_val = [np.ones((1, 4096))* 0.001]
    biases_val.append(np.ones((1, 4096)) * 0.001)
    biases_val.append(np.ones((1, n_class)) * 0.001)
    y_val = np.zeros((10, n_class))

    ctx = ndarray.gpu(0)
    for i in range(16):
        filters_val[i] = ndarray.array(filters_val[i], ctx)

    # conv 1
    conv1_1 = ad.convolution_2d_forward_op(inputs, filters1_1, "NCHW", "SAME", 1, 1)
    bn1_1 = ad.bn_forward_op(conv1_1, "NCHW", "pre_activation")
    act1_1 = ad.activation_forward_op(bn1_1, "NCHW", "relu")

    conv1_2 = ad.convolution_2d_forward_op(act1_1, filters1_2, "NCHW", "SAME", 1, 1)
    bn1_2 = ad.bn_forward_op(conv1_2, "NCHW", "pre_activation")
    act1_2 = ad.activation_forward_op(bn1_2, "NCHW", "relu")
    pool1 = ad.pooling_2d_forward_op(act1_2, "NCHW", "max", 0, 0, 2, 2, 2, 2)

    # conv 2
    conv2_1 = ad.convolution_2d_forward_op(pool1, filters2_1, "NCHW", "SAME", 1, 1)
    bn2_1 = ad.bn_forward_op(conv2_1, "NCHW", "pre_activation")
    act2_1 = ad.activation_forward_op(bn2_1, "NCHW", "relu")
    conv2_2 = ad.convolution_2d_forward_op(act2_1, filters2_2, "NCHW", "SAME", 1, 1)
    bn2_2 = ad.bn_forward_op(conv2_2, "NCHW", "pre_activation")
    act2_2 = ad.activation_forward_op(bn2_2, "NCHW", "relu")
    pool2 = ad.pooling_2d_forward_op(act2_2, "NCHW", "max", 0, 0, 2, 2, 2, 2)

    # conv 3
    conv3_1 = ad.convolution_2d_forward_op(pool2, filters3_1, "NCHW", "SAME", 1, 1)
    bn3_1 = ad.bn_forward_op(conv3_1, "NCHW", "pre_activation")
    act3_1 = ad.activation_forward_op(bn3_1, "NCHW", "relu")
    conv3_2 = ad.convolution_2d_forward_op(act3_1, filters3_2, "NCHW", "SAME", 1, 1)
    bn3_2 = ad.bn_forward_op(conv3_2, "NCHW", "pre_activation")
    act3_2 = ad.activation_forward_op(bn3_2, "NCHW", "relu")
    conv3_3 = ad.convolution_2d_forward_op(act3_2, filters3_3, "NCHW", "SAME", 1, 1)
    bn3_3 = ad.bn_forward_op(conv3_3, "NCHW", "pre_activation")
    act3_3 = ad.activation_forward_op(bn3_3, "NCHW", "relu")
    pool3 = ad.pooling_2d_forward_op(act3_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

    # conv 4
    conv4_1 = ad.convolution_2d_forward_op(pool3, filters4_1, "NCHW", "SAME", 1, 1)
    bn4_1 = ad.bn_forward_op(conv4_1, "NCHW", "pre_activation")
    act4_1 = ad.activation_forward_op(bn4_1, "NCHW", "relu")
    conv4_2 = ad.convolution_2d_forward_op(act4_1, filters4_2, "NCHW", "SAME", 1, 1)
    bn4_2 = ad.bn_forward_op(conv4_2, "NCHW", "pre_activation")
    act4_2 = ad.activation_forward_op(bn4_2, "NCHW", "relu")
    conv4_3 = ad.convolution_2d_forward_op(act4_2, filters4_3, "NCHW", "SAME", 1, 1)
    bn4_3 = ad.bn_forward_op(conv4_3, "NCHW", "pre_activation")
    act4_3 = ad.activation_forward_op(bn4_3, "NCHW", "relu")
    pool4 = ad.pooling_2d_forward_op(act4_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

    # conv 5
    conv5_1 = ad.convolution_2d_forward_op(pool4, filters5_1, "NCHW", "SAME", 1, 1)
    bn5_1 = ad.bn_forward_op(conv5_1, "NCHW", "pre_activation")
    act5_1 = ad.activation_forward_op(bn5_1, "NCHW", "relu")
    conv5_2 = ad.convolution_2d_forward_op(act5_1, filters5_2, "NCHW", "SAME", 1, 1)
    bn5_2 = ad.bn_forward_op(conv5_2, "NCHW", "pre_activation")
    act5_2 = ad.activation_forward_op(bn5_2, "NCHW", "relu")
    conv5_3 = ad.convolution_2d_forward_op(act5_2, filters5_3, "NCHW", "SAME", 1, 1)
    bn5_3 = ad.bn_forward_op(conv5_3, "NCHW", "pre_activation")
    act5_3 = ad.activation_forward_op(bn5_3, "NCHW", "relu")
    pool5 = ad.pooling_2d_forward_op(act5_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

    # fc6
    pool5_flat = ad.flatten_op(pool5)
    mul6 = ad.matmul_op(pool5_flat, filters6)
    add6 = ad.add_op(mul6, biases6)
    bn6 = ad.fullybn_forward_op(add6, "NCHW")
    fc6 = ad.fullyactivation_forward_op(bn6, "NCHW", "relu")
    drop6 = ad.fullydropout_forward_op(fc6, "NCHW", 0.5)

    # fc7
    mul7 = ad.matmul_op(drop6, filters7)
    add7 = ad.add_op(mul7, biases7)
    bn7 = ad.fullybn_forward_op(add7, "NCHW")
    fc7 = ad.fullyactivation_forward_op(bn7, "NCHW", "relu")
    drop7 = ad.fullydropout_forward_op(fc7, "NCHW", 0.5)

    #fc8
    mul8 = ad.matmul_op(drop7, filters8)
    add8 = ad.add_op(mul8, biases8)
    fc8 = ad.fullyactivation_forward_op(add8, "NCHW", "softmax")

    loss = ad.l2loss_op(fc8, y_)

    grad = ad.gradients(loss, [filters1_1, filters1_2, filters2_1, filters2_2, filters3_1, filters3_2, filters3_3
                                , filters4_1, filters4_2, filters4_3, filters5_1, filters5_2, filters5_3
                                , filters6, filters7])
    executor = ad.Executor([grad[0], grad[1], grad[2], grad[3], grad[4], grad[5], grad[6], grad[7], grad[8], grad[9]
                               , grad[10], grad[11], grad[12], grad[13], grad[14], loss, y_], ctx=ctx)

    aph = 1.0e-6
    for i in range(20):

        select = random.randint(0, n-1)
        tmp_x_val = x_val[select]
        tmp_x_val = np.expand_dims(tmp_x_val, 0)
        tmp_y_val = y_val[select]
        tmp_y_val = np.expand_dims(tmp_y_val, 0)
        grad_val = executor.run(
            feed_dict={inputs: tmp_x_val, y_: tmp_y_val
                        , filters1_1: filters_val[0], filters1_2: filters_val[1], filters2_1: filters_val[2], filters2_2: filters_val[3]
                        , filters3_1: filters_val[4], filters3_2: filters_val[5], filters3_3: filters_val[6]
                        , filters4_1: filters_val[7], filters4_2: filters_val[8], filters4_3: filters_val[9]
                        , filters5_1: filters_val[10], filters5_2: filters_val[11], filters5_3: filters_val[12]
                        , filters6: filters_val[13], filters7: filters_val[14], filters8: filters_val[15]
                        , biases6: biases_val[0], biases7: biases_val[1], biases8: biases_val[2]})


        for i in range(14):
            sgd_update_gpu(filters_val[i], grad_val[i], aph)

    print(filters_val[0].asnumpy())
    return filters_val

vgg16()