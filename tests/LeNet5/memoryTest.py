import numpy as np
from pycode.tinyflow import autodiff as ad
from pycode.tinyflow import gpu_op
from pycode.tinyflow import ndarray
from pycode.tinyflow import train


def test_dense():

    inputs = ad.Placeholder("inputs")
    filters = ad.Variable("filters")
    b = ad.Variable("b")
    y_ = ad.Variable(name="y_")

    # ini
    ctx = ndarray.gpu(0)
    x_val = np.linspace(0, 1000, 320000).reshape((3200, 1, 10, 10))
    filters_val = np.ones((32, 1, 5, 5)) * 0.001
    b_val = np.ones((32))
    y_val = np.zeros((5, 1))
    x_val = ndarray.array(x_val, ctx)
    filters_val = ndarray.array(filters_val, ctx)
    y_val = ndarray.array(y_val, ctx)

    # outputs = ad.convolution_2d_forward_op(inputs, filters, "NCHW", "VALID", 1, 1)
    outputs = ad.conv2withbias(inputs, filters, b, "NCHW", "VALID", 1, 1)

    aph = 0.001
    t = train.Adam_minimize(outputs, aph)
    # outputs_pool = ad.pooling_2d_forward_op(outputs, "NCHW", "max", 0, 0, 1, 1, 2, 2)
    #outputs_relu = ad.activation_forward_op(outputs, "NCHW", "relu")
    #executor = train.TrainExecutor([outputs], ctx=ctx)
    t.init_Variable({filters: filters_val, b: b_val})

    for i in range(100000):
        if i % 100 ==0:
            print(i)
        loss_val = t.run(feed_dict={inputs: x_val, b: b_val})

    print(loss_val[0].asnumpy())

# test_dense()

def test_pool():

    inputs = ad.Placeholder("inputs")
    filters = ad.Variable("filters")
    b = ad.Variable("b")
    y_ = ad.Variable(name="y_")

    # ini
    ctx = ndarray.gpu(0)
    x_val = np.linspace(0, 1000, 320000).reshape((3200, 1, 10, 10))
    filters_val = np.ones((32, 1, 5, 5)) * 0.001
    b_val = np.ones((32))
    y_val = np.zeros((5, 1))
    x_val = ndarray.array(x_val, ctx)
    filters_val = ndarray.array(filters_val, ctx)
    y_val = ndarray.array(y_val, ctx)

    # outputs = ad.convolution_2d_forward_op(inputs, filters, "NCHW", "VALID", 1, 1)
    outputs = ad.conv2withbias(inputs, filters, b, "NCHW", "VALID", 1, 1)
    outputs_pool = ad.pooling_2d_forward_op(outputs, "NCHW", "max", 0, 0, 1, 1, 2, 2)

    aph = 0.001
    t = train.Adam_minimize(outputs_pool, aph)

    #outputs_relu = ad.activation_forward_op(outputs, "NCHW", "relu")
    #executor = train.TrainExecutor([outputs], ctx=ctx)
    t.init_Variable({filters: filters_val, b: b_val})

    for i in range(100000):
        if i % 100 ==0:
            print(i)
        loss_val = t.run(feed_dict={inputs: x_val, b: b_val})

    print(loss_val[0].asnumpy())

# test_pool()

def test_bn():

    inputs = ad.Placeholder("inputs")
    filters = ad.Variable("filters")
    b = ad.Variable("b")
    y_ = ad.Variable(name="y_")

    # ini
    ctx = ndarray.gpu(0)
    x_val = np.linspace(0, 1000, 320000).reshape((3200, 1, 10, 10))
    filters_val = np.ones((32, 1, 5, 5)) * 0.001
    b_val = np.ones((32))
    y_val = np.zeros((5, 1))
    x_val = ndarray.array(x_val, ctx)
    filters_val = ndarray.array(filters_val, ctx)
    y_val = ndarray.array(y_val, ctx)

    # outputs = ad.convolution_2d_forward_op(inputs, filters, "NCHW", "VALID", 1, 1)
    outputs = ad.conv2withbias(inputs, filters, b, "NCHW", "VALID", 1, 1)
    outputs_pool = ad.pooling_2d_forward_op(outputs, "NCHW", "max", 0, 0, 1, 1, 2, 2)
    outputs_bn = ad.bn_forward_op(outputs_pool, "NCHW", "pre_activation")

    aph = 0.001
    t = train.Adam_minimize(outputs_bn, aph)

    #outputs_relu = ad.activation_forward_op(outputs, "NCHW", "relu")
    #executor = train.TrainExecutor([outputs], ctx=ctx)
    t.init_Variable({filters: filters_val, b: b_val})

    for i in range(100000):
        if i % 100 ==0:
            print(i)
        loss_val = t.run(feed_dict={inputs: x_val, b: b_val})

    print(loss_val[0].asnumpy())

# test_bn()

def test_flat():

    inputs = ad.Placeholder("inputs")
    filters = ad.Variable("filters")
    b = ad.Variable("b")
    y_ = ad.Variable(name="y_")

    # ini
    ctx = ndarray.gpu(0)
    x_val = np.linspace(0, 1000, 320000).reshape((3200, 1, 10, 10))
    filters_val = np.ones((32, 1, 5, 5)) * 0.001
    b_val = np.ones((32))
    y_val = np.zeros((5, 1))
    x_val = ndarray.array(x_val, ctx)
    filters_val = ndarray.array(filters_val, ctx)
    y_val = ndarray.array(y_val, ctx)

    # outputs = ad.convolution_2d_forward_op(inputs, filters, "NCHW", "VALID", 1, 1)
    outputs = ad.conv2withbias(inputs, filters, b, "NCHW", "VALID", 1, 1)
    outputs_pool = ad.pooling_2d_forward_op(outputs, "NCHW", "max", 0, 0, 1, 1, 2, 2)
    outputs_flat = ad.flatten_op(outputs_pool)

    aph = 0.001
    t = train.Adam_minimize(outputs_flat, aph)

    #outputs_relu = ad.activation_forward_op(outputs, "NCHW", "relu")
    #executor = train.TrainExecutor([outputs], ctx=ctx)
    t.init_Variable({filters: filters_val, b: b_val})

    for i in range(100000):
        if i % 100 ==0:
            print(i)
        loss_val = t.run(feed_dict={inputs: x_val, b: b_val})

    print(loss_val[0].asnumpy())

# test_flat()

def test_bnfully():

    inputs = ad.Placeholder("inputs")
    filters = ad.Variable("filters")
    b = ad.Variable("b")
    y_ = ad.Variable(name="y_")

    # ini
    ctx = ndarray.gpu(0)
    x_val = np.linspace(0, 1000, 320000).reshape((3200, 1, 10, 10))
    filters_val = np.ones((32, 1, 5, 5)) * 0.001
    b_val = np.ones((32))
    y_val = np.zeros((5, 1))
    x_val = ndarray.array(x_val, ctx)
    filters_val = ndarray.array(filters_val, ctx)
    y_val = ndarray.array(y_val, ctx)

    # outputs = ad.convolution_2d_forward_op(inputs, filters, "NCHW", "VALID", 1, 1)
    outputs = ad.conv2withbias(inputs, filters, b, "NCHW", "VALID", 1, 1)
    outputs_pool = ad.pooling_2d_forward_op(outputs, "NCHW", "max", 0, 0, 1, 1, 2, 2)
    outputs_flat = ad.flatten_op(outputs_pool)
    outputs_bn = ad.fullybn_forward_op(outputs_flat, "NCHW")

    aph = 0.001
    t = train.Adam_minimize(outputs_bn, aph)

    #outputs_relu = ad.activation_forward_op(outputs, "NCHW", "relu")
    #executor = train.TrainExecutor([outputs], ctx=ctx)
    t.init_Variable({filters: filters_val, b: b_val})

    for i in range(100000):
        if i % 100 ==0:
            print(i)
        loss_val = t.run(feed_dict={inputs: x_val, b: b_val})

    print(loss_val[0].asnumpy())

# test_bnfully()

def test_matmul():

    inputs = ad.Placeholder("inputs")
    w = ad.Variable("w")
    b = ad.Variable("b")
    y_ = ad.Variable(name="y_")

    # ini
    ctx = ndarray.gpu(0)
    x_val = np.linspace(0, 1000, 9000000).reshape((3000, 3000))
    w_val = np.linspace(0, 1000, 9000000).reshape((3000, 3000))
    b_val = np.ones((32))
    y_val = np.zeros((5, 1))
    x_val = ndarray.array(x_val, ctx)
    w_val = ndarray.array(w_val, ctx)
    y_val = ndarray.array(y_val, ctx)

    xw = ad.matmul_op(inputs, w)

    aph = 0.001
    t = train.Adam_minimize(xw, aph)

    #outputs_relu = ad.activation_forward_op(outputs, "NCHW", "relu")
    #executor = train.TrainExecutor([outputs], ctx=ctx)
    t.init_Variable({w: w_val, b: b_val})

    for i in range(100000):
        if i % 100 ==0:
            print(i)
        loss_val = t.run(feed_dict={inputs: x_val, b: b_val})

    print(loss_val[0].asnumpy())

test_matmul()