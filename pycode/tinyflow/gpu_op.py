from __future__ import absolute_import

import ctypes
from ._base import _LIB,c_array
from . import ndarray as _nd


def array_set(arr, value):
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuArraySet(arr.handle, ctypes.c_float(value))


def broadcast_to(in_arr, out_arr, type):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    if type == "NHWC":
        _LIB.DLGpuBroadcastTo0(in_arr.handle, out_arr.handle)
    elif type == "NCHW":
        _LIB.DLGpuBroadcastTo1(in_arr.handle, out_arr.handle)


def broadcast_to_backward(in_arr, out_arr, type):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    if type == "NHWC":
        _LIB.DLGpuBroadcastToBackward0(in_arr.handle, out_arr.handle)
    elif type == "NCHW":
        _LIB.DLGpuBroadcastToBackward1(in_arr.handle, out_arr.handle)


def reduce_sum_axis_zero(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuReduceSumAxisZero(in_arr.handle, out_arr.handle)


def reduce_sum(in_arr, out_arr, axis=-1):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    if axis == -1:
        _LIB.DLGpuReduceSumAll(in_arr.handle, out_arr.handle)
    else:
        _LIB.DLGpuReduceSumAxisN(in_arr.handle, out_arr.handle,ctypes.c_int(axis))

def reduce_sum_backward(dout_arr, din_arr, axis):
    assert isinstance(din_arr, _nd.NDArray)
    assert isinstance(dout_arr, _nd.NDArray)
    if axis == -1:

        _LIB.DLGpuReduceSumAllBackward(dout_arr.handle, din_arr.handle)
    else:
        _LIB.DLGpuReduceSumAxisNBackward(dout_arr.handle, din_arr.handle, ctypes.c_int(axis))


def reduce_sum_get_outshape(in_shape,axis):
    if axis == -1:
        return (1,)
    if len(in_shape)==1:
        return (1,)
    result = ()
    for i in range(len(in_shape)):
        if i != axis:
            result = result + (in_shape[i],)
    return result

def get_shape_size(shape):
    size = 1
    for i in range(len(shape)):
        size = size * shape[i]
    return size


def matrix_elementwise_add(matA, matB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuMatrixElementwiseAdd(matA.handle, matB.handle, matC.handle)


def matrix_elementwise_add_by_const(in_mat, val, out_mat):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuMatrixElementwiseAddByConst(
        in_mat.handle, ctypes.c_float(val), out_mat.handle)


def matrix_elementwise_multiply(matA, matB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuMatrixElementwiseMultiply(
        matA.handle, matB.handle, matC.handle)


def matrix_elementwise_multiply_by_const(in_mat, val, out_mat):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuMatrixMultiplyByConst(
        in_mat.handle, ctypes.c_float(val), out_mat.handle)


def matrix_multiply(matA, transA, matB, transB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuMatrixMultiply(
        matA.handle, transA, matB.handle, transB, matC.handle)


def relu(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuRelu(in_arr.handle, out_arr.handle)


def relu_gradient(in_arr, in_grad_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuReluGradient(in_arr.handle, in_grad_arr.handle, out_arr.handle)


def softmax(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuSoftmax(in_arr.handle, out_arr.handle)


def softmax_cross_entropy(in_arr_a, in_arr_b, out_arr):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(in_arr_b, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuSoftmaxCrossEntropy(
        in_arr_a.handle, in_arr_b.handle, out_arr.handle)

def cross_entropy(in_arr_a, in_arr_b, out_arr):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(in_arr_b, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuCrossEntropy(
        in_arr_a.handle, in_arr_b.handle, out_arr.handle)

def matrix_exp(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuMatrixExp(in_arr.handle, out_arr.handle)



def matrix_log(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuMatrixLog(in_arr.handle, out_arr.handle)

def matrix_reverse(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuMatrixReverse(in_arr.handle, out_arr.handle)

def matrix_pow(in_arr, val, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    val = ctypes.c_float(val)
    _LIB.DLGpuMatrixPow(in_arr.handle,val, out_arr.handle)


#��


def convolution_1d_forward(in_arr, in_filter, out_arr, cudnnlist):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_filter, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuConvolution1DForward(in_arr.handle, in_filter.handle, out_arr.handle, cudnnlist)



def convolution_2d_forward(in_arr, in_filter, out_arr,cudnnlist):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_filter, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuConvolution2DForward(in_arr.handle, in_filter.handle, out_arr.handle,cudnnlist)


def convolution_3d_forward(in_arr, in_filter, out_arr,nodelist):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_filter, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuConvolution3DForward(in_arr.handle, in_filter.handle, out_arr.handle,nodelist)


def convolution_1d_backward(in_arr,dout_arr,in_filter,in_dfilter, dinput_arr, cudnnlist):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_filter, _nd.NDArray)
    assert isinstance(dout_arr, _nd.NDArray)
    assert isinstance(in_dfilter, _nd.NDArray)
    assert isinstance(dinput_arr, _nd.NDArray)
    _LIB.DLGpuConvolution1DBackward(in_arr.handle,dout_arr.handle,in_filter.handle,in_dfilter.handle, dinput_arr.handle, cudnnlist)

def convolution_2d_backward(in_arr,dout_arr,in_filter,in_dfilter, dinput_arr, cudnnlist):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_filter, _nd.NDArray)
    assert isinstance(dout_arr, _nd.NDArray)
    assert isinstance(in_dfilter, _nd.NDArray)
    assert isinstance(dinput_arr, _nd.NDArray)
    _LIB.DLGpuConvolution2DBackward(in_arr.handle,dout_arr.handle,in_filter.handle,in_dfilter.handle, dinput_arr.handle,cudnnlist)


def convolution_3d_backward(in_arr,dout_arr,in_filter,in_dfilter, dinput_arr,cudnnlist):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_filter, _nd.NDArray)
    assert isinstance(dout_arr, _nd.NDArray)
    assert isinstance(in_dfilter, _nd.NDArray)
    assert isinstance(dinput_arr, _nd.NDArray)
    _LIB.DLGpuConvolution3DBackward(in_arr.handle,dout_arr.handle,in_filter.handle,in_dfilter.handle, dinput_arr.handle,cudnnlist)



def pooling_1d_forward(in_arr, out_arr, cudnnlist):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuPooling1DForward(in_arr.handle, out_arr.handle,cudnnlist)

def pooling_2d_forward(in_arr, out_arr,cudnnlist):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)

    _LIB.DLGpuPooling2DForward(in_arr.handle, out_arr.handle,cudnnlist)

def pooling_3d_forward(in_arr, out_arr,cudnnlist):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuPooling3DForward(in_arr.handle, out_arr.handle,cudnnlist)

def pooling_1d_backward(in_arr, out_arr,  dout_arr,din_arr,cudnnlist):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(din_arr, _nd.NDArray)
    assert isinstance(dout_arr, _nd.NDArray)
    _LIB.DLGpuPooling1DBackward(in_arr.handle, out_arr.handle,  dout_arr.handle,din_arr.handle,cudnnlist)

def pooling_2d_backward(in_arr, out_arr,  dout_arr,din_arr,cudnnlist):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(din_arr, _nd.NDArray)
    assert isinstance(dout_arr, _nd.NDArray)
    _LIB.DLGpuPooling2DBackward(in_arr.handle, out_arr.handle,  dout_arr.handle,din_arr.handle,cudnnlist)

def pooling_3d_backward(in_arr, out_arr,  dout_arr,din_arr,cudnnlist):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(din_arr, _nd.NDArray)
    assert isinstance(dout_arr, _nd.NDArray)
    _LIB.DLGpuPooling3DBackward(in_arr.handle, out_arr.handle,  dout_arr.handle,din_arr.handle,cudnnlist)


def convolution_1d_forward_get_out_shape(input_shapes,filter_shapes,dataformat,padding,v):
    output_shapes=(0,0,0)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    if padding=="SAME":
        padding=1
    elif padding=="VALID":
        padding=0
    else:
        assert 0
    assert len(input_shapes)==3 and len(filter_shapes)== 3 and len(output_shapes) == 3

    c_input_shapes = c_array(ctypes.c_int, input_shapes)

    c_filter_shapes = c_array(ctypes.c_int, filter_shapes)
    c_output_shapes = c_array(ctypes.c_int, output_shapes)

    cudnnlist = ctypes.c_int(0)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    _LIB.DLGpuConvolution1DForwardGetOutShape(c_input_shapes,c_filter_shapes,c_output_shapes,dataformat,padding,v,cudnnlist)
    return (c_output_shapes[0],c_output_shapes[1],c_output_shapes[2]),cudnnlist

def activation_forward(input,output,dataformat,activationMode):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    if activationMode == "sigmoid":
        activationMode=0
    elif activationMode == "relu":
        activationMode=1
    elif activationMode == "tanh":
        activationMode=2
    elif activationMode == "softmax":
        activationMode=3
    else:
        assert 0
    cudnnlist = ctypes.c_int(0)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    _LIB.DLGpuActivationForward(input.handle,output.handle,dataformat,activationMode,cudnnlist)
    return cudnnlist

def activation_backward(input, dinput, output, doutput,activationMode,cudnnlist):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    assert isinstance(dinput, _nd.NDArray)
    assert isinstance(doutput, _nd.NDArray)
    if activationMode == "sigmoid":
        activationMode=0
    elif activationMode == "relu":
        activationMode=1
    elif activationMode == "tanh":
        activationMode=2
    elif activationMode == "softmax":
        activationMode=3
    else:
        assert 0
    _LIB.DLGpuActivationBackward(input.handle, dinput.handle, output.handle, doutput.handle,activationMode,cudnnlist)


def pooling_1d_forward_get_out_shape(input_shapes,dataformat,poolingMode,padding_w,v,filter_w):
    output_shapes=(0,0,0)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    if poolingMode=="max":
        poolingMode=0
    elif poolingMode=="mean":
        poolingMode=1
    else:
        assert 0
    assert len(input_shapes)==3  and len(output_shapes) == 3
    c_input_shapes = c_array(ctypes.c_int, input_shapes)
    c_output_shapes = c_array(ctypes.c_int, output_shapes)
    cudnnlist = ctypes.c_int(0)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    _LIB.DLGpuPooling1DForwardGetOutShape(c_input_shapes,c_output_shapes,dataformat,poolingMode,padding_w,v,filter_w,cudnnlist)
    return (c_output_shapes[0],c_output_shapes[1],c_output_shapes[2]),cudnnlist

def convolution_2d_forward_get_out_shape(input_shapes,filter_shapes,dataformat,padding,u,v):
    output_shapes=(0,0,0,0)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    if padding=="SAME":
        padding=1
    elif padding=="VALID":
        padding=0
    else:
        assert 0
    assert len(input_shapes)==4 and len(filter_shapes)== 4 and len(output_shapes) == 4

    c_input_shapes = c_array(ctypes.c_int, input_shapes)

    c_filter_shapes = c_array(ctypes.c_int, filter_shapes)
    c_output_shapes = c_array(ctypes.c_int, output_shapes)

    cudnnlist = ctypes.c_int(0)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)

    _LIB.DLGpuConvolution2DForwardGetOutShape(c_input_shapes,c_filter_shapes,c_output_shapes,dataformat,padding,u,v,cudnnlist)
    return (c_output_shapes[0],c_output_shapes[1],c_output_shapes[2],c_output_shapes[3]), cudnnlist

# def test(y):
#     _LIB.Test(y)


def convolution_3d_forward_get_out_shape(input_shapes,filter_shapes,dataformat,padding,s1,s2,s3):
    output_shapes=(0,0,0,0,0)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    if padding=="SAME":
        padding=1
    elif padding=="VALID":
        padding=0
    else:
        assert 0
    assert len(input_shapes)==5 and len(filter_shapes)== 5 and len(output_shapes) == 5

    c_input_shapes = c_array(ctypes.c_int, input_shapes)

    c_filter_shapes = c_array(ctypes.c_int, filter_shapes)
    c_output_shapes = c_array(ctypes.c_int, output_shapes)
    cudnnlist = ctypes.c_int(0)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    _LIB.DLGpuConvolution3DForwardGetOutShape(c_input_shapes,c_filter_shapes,c_output_shapes,dataformat,padding,s1,s2,s3,cudnnlist)
    return (c_output_shapes[0],c_output_shapes[1],c_output_shapes[2],c_output_shapes[3],c_output_shapes[4]),cudnnlist


def pooling_2d_forward_get_out_shape(input_shapes,dataformat,poolingMode,padding_h,padding_w,u,v,filter_h,filter_w):
    output_shapes=(0,0,0,0)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    if poolingMode=="max":
        poolingMode=0
    elif poolingMode=="mean":
        poolingMode=1
    else:
        assert 0
    assert len(input_shapes)==4  and len(output_shapes) == 4
    c_input_shapes = c_array(ctypes.c_int, input_shapes)
    c_output_shapes = c_array(ctypes.c_int, output_shapes)

    cudnnlist = ctypes.c_int(0)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)

    _LIB.DLGpuPooling2DForwardGetOutShape(c_input_shapes,c_output_shapes,dataformat,poolingMode,padding_h,padding_w,u,v,filter_h,filter_w,cudnnlist)
    return (c_output_shapes[0],c_output_shapes[1],c_output_shapes[2],c_output_shapes[3]),cudnnlist


def pooling_3d_forward_get_out_shape(input_shapes,dataformat,poolingMode,padding1,padding2,padding3,s1,s2,s3,filter1,filter2,filter3):
    output_shapes=(0,0,0,0,0)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    if poolingMode=="max":
        poolingMode=0
    elif poolingMode=="mean":
        poolingMode=1
    else:
        assert 0
    assert len(input_shapes)==5  and len(output_shapes) == 5
    c_input_shapes = c_array(ctypes.c_int, input_shapes)
    c_output_shapes = c_array(ctypes.c_int, output_shapes)

    cudnnlist = ctypes.c_int(0)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)

    _LIB.DLGpuPooling3DForwardGetOutShape(c_input_shapes,c_output_shapes,dataformat,poolingMode,padding1,padding2,padding3,s1,s2,s3,filter1,filter2,filter3,cudnnlist)

    return (c_output_shapes[0],c_output_shapes[1],c_output_shapes[2],c_output_shapes[3],c_output_shapes[4]),cudnnlist


def dropout_forward(input,output,dataformat,dropout,seed):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    assert seed>=0
    dropout = ctypes.c_float(dropout)
    seed = ctypes.c_int(seed)
    reserveSpace_p = ctypes.c_int(0)
    reserveSpace_p = ctypes.pointer(reserveSpace_p)
    reserveSpace_p = ctypes.pointer(reserveSpace_p)

    cudnnlist = ctypes.c_int(0)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    _LIB.DLGpuDropoutForward(input.handle,output.handle,dataformat,dropout,seed,reserveSpace_p,cudnnlist)

    return reserveSpace_p,cudnnlist


def dropout_backward(doutput,dinput,reserveSpace_p,cudnnlist):
    assert isinstance(dinput, _nd.NDArray)
    assert isinstance(doutput, _nd.NDArray)

    _LIB.DLGpuDropoutBackward(doutput.handle,dinput.handle,reserveSpace_p,cudnnlist)

def cross_entropy(in_arr_a, in_arr_b, out_arr):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(in_arr_b, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuCrossEntropy(
        in_arr_a.handle, in_arr_b.handle, out_arr.handle)
def L1loss(in_arr_a, in_arr_b, out_arr):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(in_arr_b, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuL1loss(
        in_arr_a.handle, in_arr_b.handle, out_arr.handle)
def L2loss(in_arr_a, in_arr_b, out_arr):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(in_arr_b, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuL2loss(
        in_arr_a.handle, in_arr_b.handle, out_arr.handle)
def l1loss_gradient(in_arr,in_arr1, in_grad_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_arr1, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuL1LossGradient(in_arr.handle, in_arr1.handle,in_grad_arr.handle, out_arr.handle)

def l2loss_gradient(in_arr,in_arr1, in_grad_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_arr1, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuL2LossGradient(in_arr.handle, in_arr1.handle,in_grad_arr.handle, out_arr.handle)
def L1regular(in_arr_a, out_arr):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuL1regular(
        in_arr_a.handle, out_arr.handle)
def L2regular(in_arr_a,  out_arr):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuL2regular(
        in_arr_a.handle, out_arr.handle)
def l1regular_gradient(in_arr,in_grad_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuL1regularGradient(in_arr.handle,in_grad_arr.handle, out_arr.handle)

def l2regular_gradient(in_arr,in_grad_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuL2regularGradient(in_arr.handle,in_grad_arr.handle, out_arr.handle)
def concat_forward(in_arr_a, in_arr_b, out_arr):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(in_arr_b, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuConcatForward(
        in_arr_a.handle, in_arr_b.handle, out_arr.handle)
def concat_backward(in_arr_a, in_arr_b, out_arr,din_arr_a, din_arr_b):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(in_arr_b, _nd.NDArray)
    assert isinstance(din_arr_a, _nd.NDArray)
    assert isinstance(din_arr_b, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuConcatBackward(
        in_arr_a.handle, in_arr_b.handle, out_arr.handle,din_arr_a.handle, din_arr_b.handle, )

def bn_forward(input,output,dataformat,batchNormMode,n):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    if batchNormMode=="pre_activation":
        batchNormMode=0
    elif batchNormMode=="spatial":
        batchNormMode=1
    else:
        assert 0
    mean_p = ctypes.c_int(0)
    mean_p = ctypes.pointer(mean_p)
    mean_p = ctypes.pointer(mean_p)
    var_p = ctypes.c_int(0)
    var_p = ctypes.pointer(var_p)
    var_p = ctypes.pointer(var_p)
    n = ctypes.c_int(n)
    cudnnlist = ctypes.c_int(0)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    _LIB.DLGpuBatchNormalizationForward(input.handle,output.handle,dataformat,batchNormMode,n,mean_p,var_p,cudnnlist)
    return mean_p,var_p,cudnnlist


def bn_backward(input,doutput,dinput,batchNormMode,mean_p,var_p,cudnnlist):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(dinput, _nd.NDArray)
    assert isinstance(doutput, _nd.NDArray)
    if batchNormMode=="pre_activation":
        batchNormMode=0
    elif batchNormMode=="spatial":
        batchNormMode=1
    else:
        assert 0
    _LIB.DLGpuBatchNormalizationBackward(input.handle,doutput.handle,dinput.handle,batchNormMode, mean_p,var_p,cudnnlist)


def adam_compute(output, m, v, b1t, b2t, e, learning_rate):
    assert isinstance(output, _nd.NDArray)
    assert isinstance(m, _nd.NDArray)
    assert isinstance(v, _nd.NDArray)
    b1t = ctypes.c_float(b1t)
    b2t = ctypes.c_float(b2t)
    e = ctypes.c_float(e)
    learning_rate = ctypes.c_float(learning_rate)
    _LIB.DLGpuAdam(output.handle, m.handle, v.handle, b1t, b2t, e, learning_rate)


def adam_mv(m,v, g, b1,b2):
    assert isinstance(m, _nd.NDArray)
    assert isinstance(v, _nd.NDArray)
    assert isinstance(g, _nd.NDArray)
    b1 = ctypes.c_float(b1)
    b2 = ctypes.c_float(b2)
    _LIB.DLGpuAdam_mv(m.handle,v.handle, g.handle, b1,b2)






def sgd_update(output, g, learning_rate):
    assert isinstance(output, _nd.NDArray)
    assert isinstance(g, _nd.NDArray)
    learning_rate = ctypes.c_float(learning_rate)
    _LIB.DLGpuSgdUpdate(output.handle, g.handle, learning_rate)


def get_index_to_VaribaleNumber_cuda_pointer(index_to_Variable, prefix_list, index_count):
    index_count = ctypes.c_int(index_count)
    index_to_Variable = c_array(ctypes.c_int,index_to_Variable)
    prefix_list = c_array(ctypes.c_int,prefix_list)
    return _LIB.DLGpuGetIndextoVaribaleNumberCudaPointer(index_to_Variable, prefix_list, index_count)

def get_index_to_VaribaleNumber_cuda_pointer(index_to_Variable, prefix_list, index_count):
    index_count = ctypes.c_int(index_count)

    index_to_Variable = c_array(ctypes.c_int,index_to_Variable)
    prefix_list = c_array(ctypes.c_int,prefix_list)

    cudnnlist = ctypes.c_int(0)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)

    _LIB.DLGpuGetIndextoVaribaleNumberCudaPointer(index_to_Variable, prefix_list, index_count,cudnnlist)

    return cudnnlist


def get_n2_cuda_pointer(output, g, number):
    output = c_array(ctypes.c_void_p, output)
    g = c_array(ctypes.c_void_p, g)
    number = ctypes.c_int(number)

    cudnnlist = ctypes.c_int(0)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    _LIB.DLGpuGetN2CudaPointer(output, g, number,cudnnlist)
    return cudnnlist


def get_n4_cuda_pointer(output, m, v, g, number):
    output = c_array(ctypes.c_void_p, output)
    m = c_array(ctypes.c_void_p, m)
    v = c_array(ctypes.c_void_p, v)
    g = c_array(ctypes.c_void_p, g)
    number = ctypes.c_int(number)
    cudnnlist = ctypes.c_int(0)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    cudnnlist = ctypes.pointer(cudnnlist)
    _LIB.DLGpuGetN4CudaPointer(output, m, v, g, number,cudnnlist)
    return cudnnlist

def sgd_compute_o(n2list,indexinfolist,count, learning_rate):


    count = ctypes.c_int(count)
    learning_rate = ctypes.c_float(learning_rate)

    _LIB.DLGpuSgd_o(n2list, indexinfolist,count, learning_rate)

def adam_compute_o(n4list,indexinfolist,count, b1, b2, b1t, b2t, e, learning_rate):


    count = ctypes.c_int(count)
    b1 = ctypes.c_float(b1)
    b2 = ctypes.c_float(b2)
    b1t = ctypes.c_float(b1t)
    b2t = ctypes.c_float(b2t)
    e = ctypes.c_float(e)
    learning_rate = ctypes.c_float(learning_rate)

    _LIB.DLGpuAdam_o(n4list, indexinfolist,count, b1, b2 , b1t, b2t, e, learning_rate)








