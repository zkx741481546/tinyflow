from pycode.tinyflow import autodiff as ad
import numpy as np
from pycode.tinyflow import ndarray

def test_identity():
    x2 = ad.Variable(name="x2")
    y = x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


def test_add_by_const():
    x2 = ad.Variable(name="x2")
    y = 5 + x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val + 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


def test_mul_by_const():
    x2 = ad.Variable(name="x2")
    y = 5 * x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val * 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val) * 5)


def test_add_two_vars():
    x2 = ad.Variable(name="x2")
    x3 = ad.Variable(name="x3")
    y = x2 + x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val + x3_val)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))
    assert np.array_equal(grad_x3_val, np.ones_like(x3_val))


def test_mul_two_vars():
    x2 = ad.Variable(name="x2")
    x3 = ad.Variable(name="x3")
    y = x2 * x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val * x3_val)
    assert np.array_equal(grad_x2_val, x3_val)
    assert np.array_equal(grad_x3_val, x2_val)


def test_add_mul_mix_1():
    x1 = ad.Variable(name="x1")
    x2 = ad.Variable(name="x2")
    x3 = ad.Variable(name="x3")
    y = x1 + x2 * x3 * x1

    grad_x1, grad_x2, grad_x3 = ad.gradients(y, [x1, x2, x3])

    executor = ad.Executor([y, grad_x1, grad_x2, grad_x3])
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x1: x1_val, x2: x2_val, x3: x3_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x1_val + x2_val * x3_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val) + x2_val * x3_val)
    assert np.array_equal(grad_x2_val, x3_val * x1_val)
    assert np.array_equal(grad_x3_val, x2_val * x1_val)


def test_add_mul_mix_2():
    x1 = ad.Variable(name="x1")
    x2 = ad.Variable(name="x2")
    x3 = ad.Variable(name="x3")
    x4 = ad.Variable(name="x4")
    y = x1 + x2 * x3 * x4

    grad_x1, grad_x2, grad_x3, grad_x4 = ad.gradients(y, [x1, x2, x3, x4])

    executor = ad.Executor([y, grad_x1, grad_x2, grad_x3, grad_x4])
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    x4_val = 4 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val, grad_x3_val, grad_x4_val = executor.run(
        feed_dict={x1: x1_val, x2: x2_val, x3: x3_val, x4: x4_val}
    )

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x1_val + x2_val * x3_val * x4_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val))
    assert np.array_equal(grad_x2_val, x3_val * x4_val)
    assert np.array_equal(grad_x3_val, x2_val * x4_val)
    assert np.array_equal(grad_x4_val, x2_val * x3_val)


def test_add_mul_mix_3():
    x2 = ad.Variable(name="x2")
    x3 = ad.Variable(name="x3")
    z = x2 * x2 + x2 + x3 + 3
    y = z * z + x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

    z_val = x2_val * x2_val + x2_val + x3_val + 3
    expected_yval = z_val * z_val + x3_val
    expected_grad_x2_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) * (2 * x2_val + 1)
    expected_grad_x3_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) + 1
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)


def test_grad_of_grad():
    x2 = ad.Variable(name="x2")
    x3 = ad.Variable(name="x3")
    y = x2 * x2 + x2 * x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
    grad_x2_x2, grad_x2_x3 = ad.gradients(grad_x2, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3, grad_x2_x2, grad_x2_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val, grad_x2_x2_val, grad_x2_x3_val = executor.run(
        feed_dict={x2: x2_val, x3: x3_val}
    )

    expected_yval = x2_val * x2_val + x2_val * x3_val
    expected_grad_x2_val = 2 * x2_val + x3_val
    expected_grad_x3_val = x2_val
    expected_grad_x2_x2_val = 2 * np.ones_like(x2_val)
    expected_grad_x2_x3_val = 1 * np.ones_like(x2_val)

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)
    assert np.array_equal(grad_x2_x2_val, expected_grad_x2_x2_val)
    assert np.array_equal(grad_x2_x3_val, expected_grad_x2_x3_val)


def test_matmul_two_vars():
    x2 = ad.Variable(name="x2")
    x3 = ad.Variable(name="x3")
    y = ad.matmul_op(x2, x3)

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
    x3_val = np.array([[7, 8, 9], [10, 11, 12]])  # 2x3

    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

    expected_yval = np.matmul(x2_val, x3_val)
    expected_grad_x2_val = np.matmul(np.ones_like(expected_yval), np.transpose(x3_val))
    expected_grad_x3_val = np.matmul(np.transpose(x2_val), np.ones_like(expected_yval))

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)


def test_exp():
    x1 = ad.Variable("x1")
    x2 = ad.exp_op(x1)
    x3 = x2 + 1
    x4 = x2 * x3

    x1_grad, = ad.gradients(x4, [x1])

    executor = ad.Executor([x4])
    x1_val = 1
    x4_val, x1_grad = executor.run(feed_dict={x1: x1_val})
    print(x4_val)
    print(x1_grad)


def test_exp_grad():
    x = ad.Variable("x")
    y = ad.exp_op(x)

    x_grad, = ad.gradients(y, [x])

    executor = ad.Executor([y, x_grad])
    x_val = 1
    y_val, x_grad_val = executor.run(feed_dict={x: x_val})
    print(y_val)
    print(x_grad_val)


def test_lr():
    W = ad.Variable(name="W")
    b = ad.Variable(name="b")
    X = ad.Variable(name="X")
    y_ = ad.Variable(name="y_")
    
    ctx = ndarray.gpu(0)
    # ini 
    x_val = np.linspace(0,1,100).reshape((100,1))
    y_val = x_val + 0.5
    W_val = np.array([[0.1]])
    b_val = np.array([0.1])
    x_val = ndarray.array(x_val, ctx)
    W_val = ndarray.array(W_val, ctx)
    b_val = ndarray.array(b_val, ctx)
    y_val = ndarray.array(y_val, ctx)
    z = ad.matmul_op(X, W)
    # z.shape = (100,1)
    # b.shape = (1,1)
    y = z + ad.broadcastto_op(b, z)
    # y = (100,1)
    y = ad.fullyactivation_forward_op(y,"NCHW","relu")
    loss = ad.matmul_op(y + (-1)*y_, y + (-1)*y_, trans_A=True) * (1/100)
    # loss = ad.softmaxcrossentropy_op(y, y_)
    grad_W, grad_b = ad.gradients(loss, [W, b])

    executor = ad.Executor([loss, grad_W, grad_b],ctx)

    aph = 1e-6

    for i in range(100):

        loss_val, grad_W_val ,grad_b_val = executor.run(feed_dict={X: x_val,b: b_val,W: W_val,y_:y_val})


        grad_W_val = grad_W_val.asnumpy()
        W_val = W_val.asnumpy()
        W_val = W_val - aph * grad_W_val
        W_val = ndarray.array(W_val, ctx)

        grad_b_val = grad_b_val.asnumpy()
        b_val = b_val.asnumpy()
        b_val = b_val - aph * grad_b_val
        b_val = ndarray.array(b_val, ctx)
        print(W_val.asnumpy(), b_val.asnumpy())
   # executor = ad.Executor([y])
   # res = executor.run(feed_dict={X: x_val,b: b_val,W: W_val})
 #   print('y_true'+str(y_val))
 #   print('y_pred'+str(res))




def test_convolution_1d_forward_op():
    inputs = ad.Variable("inputs")
    filters = ad.Variable("filters")
    y_ = ad.Variable(name="y_")

    #ini
    ctx = ndarray.gpu(0)
    x_val = np.linspace(0,100,100).reshape((5,1,20))
    filters_val = np.ones((1,1,20))*0.001
    y_val = np.zeros((5,1))
    x_val = ndarray.array(x_val,ctx)
    filters_val = ndarray.array(filters_val,ctx)
    y_val = ndarray.array(y_val, ctx)
    outputs = ad.convolution_1d_forward_op(inputs, filters, "NCHW", "VALID", 1)
    outputs_pool = ad.pooling_1d_forward_op(outputs,"NCHW","max",0,1,1)
    outputs_relu = ad.activation_forward_op(outputs_pool,"NCHW","relu")
    outputs_f = ad.flatten_op(outputs_relu)
    outputs_fu = ad.fullyactivation_forward_op(outputs_f, "NCHW", "relu")
    loss = ad.matmul_op(outputs_fu, outputs_fu, trans_A=True) *(1/5)
    grad_inputs,grad_f = ad.gradients(loss, [inputs,filters])
    executor = ad.Executor([loss, grad_f],ctx=ctx)

    aph = 1.0e-6
    for i in range(10):
        loss_val, filters_grad_val = executor.run(feed_dict={inputs: x_val,filters:filters_val})


        filters_val = filters_val.asnumpy()
        filters_grad_val = filters_grad_val.asnumpy()
        filters_val = filters_val - aph * filters_grad_val
        filters_val = ndarray.array(filters_val, ctx)

    print("loss_val:",loss_val.asnumpy())
    print("filters_val:",filters_val.asnumpy())

def test_convolution_2d_forward_op():
    inputs = ad.Variable("inputs")
    filters = ad.Variable("filters")
    y_ = ad.Variable(name="y_")


    #ini
    ctx = ndarray.gpu(0)
    x_val = np.linspace(0,100,80).reshape((5,1,4,4))
    filters_val = np.ones((1,1,3,3))*0.001
    y_val = np.zeros((5,1))
    x_val = ndarray.array(x_val,ctx)
    filters_val = ndarray.array(filters_val,ctx)
    y_val = ndarray.array(y_val, ctx)


    outputs = ad.convolution_2d_forward_op(inputs, filters, "NCHW", "VALID",1,1)
    outputs_pool = ad.pooling_2d_forward_op(outputs,"NCHW","max",0,0,1,1,2,2)
    outputs_relu = ad.activation_forward_op(outputs_pool,"NCHW","relu")
    outputs_f = ad.flatten_op(outputs_relu)
    loss = ad.matmul_op(outputs_f, outputs_f, trans_A=True) *(1/5)
    grad_inputs,grad_f = ad.gradients(loss, [inputs,filters])
    executor = ad.Executor([loss, grad_f],ctx=ctx)

    aph = 1.0e-6
    for i in range(20):
        loss_val, filters_grad_val = executor.run(feed_dict={inputs: x_val,filters:filters_val})


        filters_val = filters_val.asnumpy()
        filters_grad_val = filters_grad_val.asnumpy()
        filters_val = filters_val - aph * filters_grad_val
        filters_val = ndarray.array(filters_val, ctx)
    print("loss_val:",loss_val.asnumpy())
    print("filters_val:",filters_val.asnumpy())

def test_convolution_3d_forward_op():
    inputs = ad.Variable("inputs")
    filters = ad.Variable("filters")
    y_ = ad.Variable(name="y_")


    #ini
    ctx = ndarray.gpu(0)
    x_val = np.linspace(0,100,135).reshape((5,1,3,3,3))
    filters_val = np.ones((1,1,2,2,2))*0.001
    y_val = np.zeros((5,1))
    x_val = ndarray.array(x_val,ctx)
    filters_val = ndarray.array(filters_val,ctx)
    y_val = ndarray.array(y_val, ctx)


    outputs = ad.convolution_3d_forward_op(inputs, filters, "NCHW", "VALID",1,1,1)
    outputs_pool = ad.pooling_3d_forward_op(outputs,"NCHW","max",0,0,0,1,1,1,2,2,2)
    outputs_relu = ad.activation_forward_op(outputs_pool,"NCHW","relu")
    outputs_dro = ad.dropout_forward_op(outputs_relu,"NCHW",0.5,0)
    outputs_f = ad.flatten_op(outputs_dro)
    loss = ad.matmul_op(outputs_f, outputs_f, trans_A=True) *(1/5)
    grad_inputs,grad_f = ad.gradients(loss, [inputs,filters])
    executor = ad.Executor([loss, grad_f],ctx=ctx)

    aph = 1.0e-6
    for i in range(20):
        loss_val, filters_grad_val = executor.run(feed_dict={inputs: x_val,filters:filters_val})


        filters_val = filters_val.asnumpy()
        filters_grad_val = filters_grad_val.asnumpy()
        filters_val = filters_val - aph * filters_grad_val
        filters_val = ndarray.array(filters_val, ctx)

    print("loss_val:",loss_val.asnumpy())
    print("filters_val:",filters_val.asnumpy())


def test_sigmoid_conv_1d():
    inputs = ad.Variable("inputs")
    filters = ad.Variable("filters")
    y_ = ad.Variable(name="y_")

    # ini
    ctx = ndarray.gpu(0)
    x_val = np.linspace(0, 100, 80).reshape((5, 1, 4, 4))
    filters_val = np.ones((1, 1, 3, 3)) * 0.001
    y_val = np.zeros((5, 1))
    x_val = ndarray.array(x_val, ctx)
    filters_val = ndarray.array(filters_val, ctx)
    y_val = ndarray.array(y_val, ctx)

    outputs = ad.convolution_2d_forward_op(inputs, filters, "NCHW", "VALID", 1, 1)
   # outputs_pool = ad.pooling_2d_forward_op(outputs, "NCHW", "max", 0, 0, 1, 1, 2, 2)
    outputs_relu = ad.activation_forward_op(outputs, "NCHW", "relu")
    executor = ad.Executor([outputs_relu],ctx=ctx)


    loss_val= executor.run(feed_dict={inputs: x_val,filters:filters_val})


    print("loss_val:",loss_val[0].asnumpy())

def test_full_forward_op():
    inputs = ad.Variable("inputs")
    filters = ad.Variable("filters")
    y_ = ad.Variable(name="y_")

    #ini
    ctx = ndarray.gpu(0)
    x_val = np.linspace(0,100,100).reshape((5,1,20))
    filters_val = np.ones((1,1,20))*0.001
    y_val = np.zeros((5,1))
    x_val = ndarray.array(x_val,ctx)
    filters_val = ndarray.array(filters_val,ctx)
    y_val = ndarray.array(y_val, ctx)
    outputs = ad.convolution_1d_forward_op(inputs, filters, "NCHW", "VALID", 1)
    outputs_pool = ad.pooling_1d_forward_op(outputs,"NCHW","max",0,1,1)
    outputs_relu = ad.activation_forward_op(outputs_pool,"NCHW","relu")
    outputs_f = ad.flatten_op(outputs_relu)
    output=ad.fullyactivation_forward_op(outputs_f,"NCHW","relu")
    loss = ad.matmul_op(output, output, trans_A=True) * (1 / 5)
    grad_f = ad.gradients(loss,[filters])#gra返回一个list
    executor = ad.Executor([grad_f[0]],ctx=ctx)
    g_val = executor.run(feed_dict={inputs: x_val,filters:filters_val})#返回一个list
    print("g_val:", g_val[0].asnumpy())




def test_exp_log_reverse_pow():
    inputs = ad.Variable("inputs")
    filters = ad.Variable("filters")
    y_ = ad.Variable(name="y_")

    # ini
    ctx = ndarray.gpu(0)
    x_val = np.linspace(0, 100, 80).reshape((5, 1, 4, 4))
    filters_val = np.ones((1, 1, 3, 3)) * 0.001
    y_val = np.zeros((5, 1))
    x_val = ndarray.array(x_val, ctx)
    filters_val = ndarray.array(filters_val, ctx)
    y_val = ndarray.array(y_val, ctx)

    #outputs = ad.exp_op(inputs)
    #outputs = ad.log_op(inputs)
    #outputs = ad.reverse_op(inputs)
    outputs = ad.pow_op(inputs,2)
    grad_out = ad.gradients(outputs, [inputs])
    executor = ad.Executor([outputs,grad_out[0]], ctx=ctx)
    result= executor.run(feed_dict={inputs: filters_val})

    print(result[0].asnumpy())
    print(result[1].asnumpy())







def test_reduce_sum():
    inputs = ad.Variable("inputs")

    ctx = ndarray.gpu(0)
    shape = (3,2,3)
    x = np.random.uniform(0, 20, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)


    outputs = ad.reduce_sum_op(inputs,1)
    f_out = ad.pow_op(outputs,2)
    grad_out = ad.gradients(f_out, [inputs])
    executor = ad.Executor([outputs,f_out , grad_out[0]], ctx=ctx)
    result = executor.run(feed_dict={inputs: arr_x})
    print(arr_x.asnumpy())
    print(result[0].asnumpy())
    print(result[1].asnumpy())
    print(result[2].asnumpy())



def test_reduce_mean():
    inputs = ad.Variable("inputs")

    ctx = ndarray.gpu(0)
    shape = (2,2,3)
    x = np.random.uniform(0, 20, shape).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)


    outputs = ad.reduce_mean_op(inputs,1)
    f_out = ad.pow_op(outputs,2)
    grad_out = ad.gradients(f_out, [inputs])
    executor = ad.Executor([outputs,f_out , grad_out[0]], ctx=ctx)
    result = executor.run(feed_dict={inputs: arr_x})
    print(arr_x.asnumpy())
    print(result[0].asnumpy())
    print(result[1].asnumpy())
    print(result[2].asnumpy())


def test_l1_l2_cross_loss():
    inputs = ad.Variable("inputs")
    filters = ad.Variable("filters")
    y_ = ad.Variable(name="y_")

    # ini
    ctx = ndarray.gpu(0)
    x_val = np.ones((5,2))*0.5
    filters_val = np.ones((2, 2, 10)) * 0.001
    y_val = np.ones((5,2))
    x_val = ndarray.array(x_val, ctx)
    filters_val = ndarray.array(filters_val, ctx)
    y_val = ndarray.array(y_val, ctx)
    # loss = ad.crossEntropy_op(inputs, y_)
    loss = ad.l1loss_op(inputs, y_)
    grad_f = ad.gradients(loss, [inputs,y_])  # gra返回一个list
    executor = ad.Executor([loss,grad_f[0],grad_f[1]], ctx=ctx)
    g_val = executor.run(feed_dict={inputs: x_val, y_: y_val})  # 返回一个list
    print("g_val:", g_val[0].asnumpy())
    print("g_val:", g_val[1].asnumpy())
    print("g_val:", g_val[2].asnumpy())
def test_l1_l2_regular():
    inputs = ad.Variable("inputs")
    filters = ad.Variable("filters")
    y_ = ad.Variable(name="y_")

    # ini
    ctx = ndarray.gpu(0)
    x_val = np.ones((5,2))*0.5
    x_val = ndarray.array(x_val, ctx)
    loss = ad.l2regular_op(inputs)
    # loss = ad.l1regular_op(inputs)
    grad_f = ad.gradients(loss, [inputs])  # gra返回一个list
    executor = ad.Executor([loss,grad_f[0]], ctx=ctx)
    g_val = executor.run(feed_dict={inputs: x_val})  # 返回一个list
    print("g_val:", g_val[0].asnumpy())
    print("g_val:", g_val[1].asnumpy())

#test_lr()
# test_identity()
# test_add_by_const()
# test_mul_by_const()
# test_add_two_vars()
# test_mul_two_vars()
# test_add_mul_mix_1()
# test_add_mul_mix_2()
# test_add_mul_mix_3()
# test_grad_of_grad()
# test_matmul_two_vars()

#test_full_forward_op()
#test_sigmoid_conv_1d()
# =============not implement yet====================
# test_exp()
# test_exp_grad()

#test_full_forward_op()
#test_full_forward_op()
#test_exp_log_reverse_pow()
#test_reduce_sum()
#test_reduce_mean()
# test_convolution_1d_forward_op()
# test_convolution_2d_forward_op()
# test_convolution_3d_forward_op()
# test_l1_l2_cross_loss()
test_l1_l2_regular()