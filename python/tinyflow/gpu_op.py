from __future__ import absolute_import

import ctypes
from ._base import _LIB
from . import ndarray as _nd


def array_set(arr, value):
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuArraySet(arr.handle, ctypes.c_float(value))


def broadcast_to(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuBroadcastTo(in_arr.handle, out_arr.handle)


def reduce_sum_axis_zero(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuReduceSumAxisZero(in_arr.handle, out_arr.handle)


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



#��


def convolution_1d_forward(in_arr, in_filter, out_arr,dataformat, padding, v):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_filter, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
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
    _LIB.DLGpuConvolution1DForward(in_arr.handle, in_filter.handle, out_arr.handle,dataformat, padding,  v)

def convolution_2d_forward(in_arr, in_filter, out_arr,dataformat, padding, u, v):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_filter, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
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
    _LIB.DLGpuConvolution2DForward(in_arr.handle, in_filter.handle, out_arr.handle,dataformat, padding, u, v)


def convolution_3d_forward(in_arr, in_filter, out_arr,dataformat, padding, s1, s2, s3):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_filter, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
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
    _LIB.DLGpuConvolution3DForward(in_arr.handle, in_filter.handle, out_arr.handle,dataformat, padding, s1, s2, s3)


def convolution_1d_backward(in_arr,dout_arr,in_filter,in_dfilter, dinput_arr,dataformat,padding, v):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_filter, _nd.NDArray)
    assert isinstance(dout_arr, _nd.NDArray)
    assert isinstance(in_dfilter, _nd.NDArray)
    assert isinstance(dinput_arr, _nd.NDArray)
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
    _LIB.DLGpuConvolution1DBackward(in_arr.handle,dout_arr.handle,in_filter.handle,in_dfilter.handle, dinput_arr.handle,dataformat,padding,  v)

def convolution_2d_backward(in_arr,dout_arr,in_filter,in_dfilter, dinput_arr,dataformat, padding, u, v):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_filter, _nd.NDArray)
    assert isinstance(dout_arr, _nd.NDArray)
    assert isinstance(in_dfilter, _nd.NDArray)
    assert isinstance(dinput_arr, _nd.NDArray)
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
    _LIB.DLGpuConvolution2DBackward(in_arr.handle,dout_arr.handle,in_filter.handle,in_dfilter.handle, dinput_arr.handle,dataformat,padding, u, v)


def convolution_3d_backward(in_arr,dout_arr,in_filter,in_dfilter, dinput_arr,dataformat,padding, s1, s2, s3):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_filter, _nd.NDArray)
    assert isinstance(dout_arr, _nd.NDArray)
    assert isinstance(in_dfilter, _nd.NDArray)
    assert isinstance(dinput_arr, _nd.NDArray)
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
    _LIB.DLGpuConvolution3DBackward(in_arr.handle,dout_arr.handle,in_filter.handle,in_dfilter.handle, dinput_arr.handle,dataformat,padding, s1, s2, s3)



def pooling_1d_forward(in_arr, out_arr, dataformat, pad_w, v, filter_w):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    _LIB.DLGpuPooling1DForward(in_arr.handle, out_arr.handle, dataformat, pad_w, v,filter_w)

def pooling_2d_forward(in_arr, out_arr,dataformat, pad_h, pad_w, u, v,filter_h,filter_w):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    _LIB.DLGpuPooling2DForward(in_arr.handle, out_arr.handle,dataformat, pad_h, pad_w, u, v,filter_h,filter_w)

def pooling_3d_forward(in_arr, out_arr,dataformat, pad1, pad2,pad3, s1, s2,s3,filter1,filter2,filter3):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    _LIB.DLGpuPooling3DForward(in_arr.handle, out_arr.handle,dataformat, pad1, pad2, pad3, s1, s2, s3,filter1,filter2,filter3)

def pooling_1d_Backward(in_arr, out_arr,  dout_arr,din_arr,dataformat,pad_w, v, filter_w):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(din_arr, _nd.NDArray)
    assert isinstance(dout_arr, _nd.NDArray)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    _LIB.DLGpuPooling1DBackward(in_arr.handle, out_arr.handle,  dout_arr.handle,din_arr.handle,dataformat,pad_w, v,filter_w)

def pooling_2d_Backward(in_arr, out_arr,  dout_arr,din_arr,dataformat,pad_h, pad_w, u, v,filter_h,filter_w):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(din_arr, _nd.NDArray)
    assert isinstance(dout_arr, _nd.NDArray)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    _LIB.DLGpuPooling2DBackward(in_arr.handle, out_arr.handle,  dout_arr.handle,din_arr.handle,dataformat,pad_h, pad_w, u, v,filter_h,filter_w)

def pooling_3d_Backward(in_arr, out_arr,  dout_arr,din_arr,dataformat,pad1, pad2,pad3, s1, s2,s3,filter1,filter2,filter3):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(din_arr, _nd.NDArray)
    assert isinstance(dout_arr, _nd.NDArray)
    if dataformat=="NCHW":
        dataformat=0
    elif dataformat=="NHWC":
        dataformat=1
    else:
        assert 0
    _LIB.DLGpuPooling3DBackward(in_arr.handle, out_arr.handle,  dout_arr.handle,din_arr.handle,dataformat,pad1, pad2, pad3, s1, s2, s3,filter1,filter2,filter3)


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
    assert len(input_shapes)==3 and len(input_shapes)== 3 and len(output_shapes) == 3
    _LIB.DLGpuConvolution1DForwardGetOutShape(input_shapes,filter_shapes,output_shapes,dataformat,padding,v)
    return output_shapes