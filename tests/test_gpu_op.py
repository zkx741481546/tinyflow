import numpy as np
from pycode.tinyflow import ndarray, gpu_op, autodiff


def test_array_set():
    ctx = ndarray.gpu(0)
    shape = (500, 200)
    # oneslike
    arr_x = ndarray.empty(shape, ctx=ctx)
    gpu_op.array_set(arr_x, 1.)
    x = arr_x.asnumpy()
    np.testing.assert_allclose(np.ones(shape), x)
    # zeroslike
    gpu_op.array_set(arr_x, 0.)
    x = arr_x.asnumpy()
    np.testing.assert_allclose(np.zeros(shape), x)


def test_broadcast_to():
    ctx = ndarray.gpu(0)
    shape = (2, 3)
    to_shape = ( 5,2,3)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(to_shape, ctx=ctx)
    gpu_op.broadcast_to(arr_x, arr_y)
    y = arr_y.asnumpy()
    print(arr_x.asnumpy())
    print(y)
    np.testing.assert_allclose(np.broadcast_to(x, to_shape), y)


def test_reduce_sum_axis_zero():
    ctx = ndarray.gpu(0)
    shape = (5)
    to_shape = (1,)
    x = np.random.uniform(0, 20, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(to_shape, ctx=ctx)
    gpu_op.reduce_sum_axis_zero(arr_x, arr_y)
    print(arr_x.asnumpy())
    print(arr_y.asnumpy())
    y = arr_y.asnumpy()
    y_ = np.sum(x, axis=0)
    for index, _ in np.ndenumerate(y):
        v = y[index]
        v_ = y_[index]
        if abs((v - v_) / v_) > 1e-4:
            print(index, v, v_)
    np.testing.assert_allclose(np.sum(x, axis=0), y, rtol=1e-5)


def test_matrix_elementwise_add():
    ctx = ndarray.gpu(0)
    shape = (500, 200)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    y = np.random.uniform(0, 10, size=shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_add(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x + y, z, rtol=1e-5)


def test_matrix_elementwise_add_by_const():
    shape = (2000, 3000)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    val = np.random.uniform(-5, 5)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_add_by_const(arr_x, val, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(x + val, y, rtol=1e-5)


def test_matrix_elementwise_multiply():
    ctx = ndarray.gpu(0)
    shape = (500, 200)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    y = np.random.uniform(0, 10, size=shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_multiply(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x * y, z, rtol=1e-5)


def test_matrix_elementwise_multiply_by_const():
    shape = (2000, 3000)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    val = np.random.uniform(-5, 5)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_multiply_by_const(arr_x, val, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(x * val, y, rtol=1e-5)


def test_matrix_multiply():
    ctx = ndarray.gpu(0)
    x = np.random.uniform(0, 10, size=(500, 700)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(700, 1000)).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((500, 1000), ctx=ctx)
    gpu_op.matrix_multiply(arr_x, False, arr_y, False, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(x, y), z, rtol=1e-5)

    x = np.random.uniform(0, 10, size=(1000, 500)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(2000, 500)).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((1000, 2000), ctx=ctx)
    gpu_op.matrix_multiply(arr_x, False, arr_y, True, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(x, np.transpose(y)), z, rtol=1e-5)
    
    x = np.random.uniform(0, 10, size=(500, 1000)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(2000, 500)).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((1000, 2000), ctx=ctx)
    gpu_op.matrix_multiply(arr_x, True, arr_y, True, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(np.transpose(x), np.transpose(y)), z,
                               rtol=1e-5)


def test_relu():
    shape = (2000, 2500)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.relu(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.maximum(x, 0).astype(np.float32), y)


def test_relu_gradient():
    shape = (2000, 2500)
    ctx = ndarray.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    grad_x = np.random.uniform(-5, 5, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_grad_x = ndarray.array(grad_x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.relu_gradient(arr_x, arr_grad_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(((x > 0) * grad_x).astype(np.float32), y)


def test_softmax():
    ctx = ndarray.gpu(0)
    shape = (400, 1000)
    x = np.random.uniform(-5, 5, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    gpu_op.softmax(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(autodiff.softmax_func(x), y, rtol=1e-5)


def test_softmax_cross_entropy():
    ctx = ndarray.gpu(0)
    shape = (400, 1000)
    y = np.random.uniform(-5, 5, shape).astype(np.float32)
    y_ = np.random.uniform(-5, 5, shape).astype(np.float32)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_y_ = ndarray.array(y_, ctx=ctx)
    arr_out = ndarray.empty((1,), ctx=ctx)
    gpu_op.softmax_cross_entropy(arr_y, arr_y_, arr_out)
    out = arr_out.asnumpy()
    # numpy calculation
    cross_entropy = np.mean(
        -np.sum(y_ * np.log(autodiff.softmax_func(y)), axis=1), keepdims=True)
    np.testing.assert_allclose(cross_entropy, out, rtol=1e-5)


#��
def test_convolution_forward():
    ctx = ndarray.gpu(0)
    in_shape = (1,1,8)
    filter_shape = (1, 1, 5)
    out_shape = (1, 1, 8)
    input_arr_np = np.arange(8).reshape(in_shape)

    dinput_arr_np = np.arange(8).reshape(in_shape)
    filter_arr_np = np.arange(5).reshape(filter_shape)
    dfilter_arr_np = np.arange(5).reshape(filter_shape)
    dinput=ndarray.array(dinput_arr_np, ctx=ctx)
    dfilter=ndarray.array(dfilter_arr_np, ctx=ctx)
    arr_in = ndarray.array(input_arr_np, ctx=ctx)
    arr_filter = ndarray.array(filter_arr_np, ctx=ctx)
    arr_out = ndarray.empty(out_shape, ctx=ctx)
    gpu_op.convolution_1d_forward(arr_in,arr_filter,arr_out,"NCHW","SAME",1)
    gpu_op.convolution_1d_backward(arr_in,arr_out,arr_filter,dfilter,dinput,"NCHW","SAME",1)
    print(arr_out.asnumpy())
    print(dfilter.asnumpy())
    print(dinput.asnumpy())
    print(arr_in.asnumpy())


def test_reshape():
    ctx = ndarray.gpu(0)
    in_shape = (5,  2)
    input_arr_np = np.arange(10).reshape(in_shape)
    arr_in = ndarray.array(input_arr_np, ctx=ctx)
    print(arr_in.asnumpy())
    arr_in = arr_in.reshape((5,1,2))
    print(arr_in.asnumpy())
    print(len(arr_in.shape))

def test_convolution_backward():
    ctx = ndarray.gpu(0)
    x_val1 = np.linspace(0, 100, 100).reshape((5, 1, 20))
    filters_val1 = np.ones((1, 1, 20)) * 0.001
    y_val = np.array([[[0.07676768]],[[0.23838384]],[[0.40000007]],[[0.5616162 ]],[[0.7232323]]])
    x_val = ndarray.array(x_val1, ctx)
    d_x_val = ndarray.array(x_val1, ctx)
    d_filters_val = ndarray.array(filters_val1, ctx)
    filters_val = ndarray.array(filters_val1, ctx)
    y_val = ndarray.array(y_val, ctx)
    gpu_op.convolution_1d_backward(x_val,y_val,filters_val,d_filters_val,d_x_val,"NCHW","VALID",1)
    print(1)
    print(d_x_val.asnumpy())
    print(d_filters_val.asnumpy())


def test_dropout():
    ctx = ndarray.gpu(0)
    in_shape = (5, 1, 2)

    input_arr_np = np.arange(10).reshape(in_shape)
    arr_in = ndarray.array(input_arr_np, ctx=ctx)
    arr_out = ndarray.empty(in_shape,ctx=ctx)
    r = gpu_op.dropout_forward(arr_in,arr_out,"NCHW",0.2,1)
    print(arr_in.asnumpy())
    print(arr_out.asnumpy())
    gpu_op.dropout_backward(arr_out, arr_in, "NCHW", 0.2,1, r)
    print(arr_in.asnumpy())

def test_reduce_sum_axis_n():
    ctx = ndarray.gpu(0)
    shape = (5)
    to_shape = (1,)
    x = np.random.uniform(0, 20, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_y = ndarray.empty(to_shape, ctx=ctx)
    gpu_op.reduce_sum(arr_x, arr_y)

    print(arr_x.shape[0])
    print(arr_y.asnumpy())



#test_reduce_sum_axis_zero()
#test_softmax_cross_entropy()
#test_reshape()
#test_reduce_sum_axis_n()
#test_convolution_backward()
#test_convolution_forward()
#test_dropout()
test_broadcast_to()