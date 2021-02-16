from pycode.tinyflow import autodiff as ad
import numpy as np
from pycode.tinyflow import ndarray

def bnt(num):
    inputs = ad.Variable("inputs")
    ctx = ndarray.cpu(0)
    x_val = np.linspace(0, num*num,num*num).reshape((num, num))
    x_val = ndarray.array(x_val, ctx=ctx)
    loss = ad.fullybn_forward_op(inputs, "NCHW")
    print(44444)
    executor = ad.Executor([loss], ctx=ctx)
    print(44444)
    g_val = executor.run(feed_dict={inputs: x_val})
    print(44444)
    print(num * num)
    return 0

def  test_bnfor():
    print(5555)
    for i in range (400):
        bnt(25*(i+1))

def ac(num):
    mode="sigmoid"
    inputs = ad.Variable("inputs")
    # ini
    ctx = ndarray.gpu(0)
    x_val = np.linspace(0, num * num*3, num * num*3).reshape((1, 3, num, num))
    x_val = ndarray.array(x_val, ctx)

    loss = ad.activation_forward_op(inputs, "NCHW", mode)
    executor = ad.Executor([loss], ctx=ctx)
    g_val = executor.run(feed_dict={inputs: x_val})
    print(num * num*3)

def  test_acfor():
    for i in range (400):
        ac(15*(i+1))
test_bnfor()