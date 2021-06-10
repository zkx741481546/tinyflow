
import numpy as np
load_list = ['convolution_2d_forward_VALID','convolution_backward_filter_2d_VALID','convolution_backward_data_2d_VALID',
'convolution_2d_forward_SAME','convolution_backward_filter_2d_SAME','convolution_backward_data_2d_SAME',
'dropout_forward','dropout_backward','bn_forward_pre_activation',
'bn_backward_pre_activation','pooling_2d_forward_max','pooling_2d_backward_max','pooling_2d_forward_mean',
'pooling_2d_backward_mean','broadcast_to_NHWC',
'broadcast_to_NCHW','reduce_sum_new_NHWC','reduce_sum_new_NCHW','cross','cross_backward','adam_mv','adam_compute',
'activation_forward_relu','activation_backward_relu','activation_forward_softmax','activation_backward_softmax',
'matrix_multiply','matrix_elementwise_add','concat_forward','concat_a_backward',
'concat_b_backward','sgd_update','matrix_elementwise_multiply_by_const','array_set']
#inputsshape是[],对应new_node.inputs = [node_A, node_B]的shape
def getinputsofmodel(node,inputsshape):
    if node.name == "Convolution2DForward":
        if node.padding == "VALID":
            opname = 'convolution_2d_forward_VALID'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], inputsshape[1][0],
                             inputsshape[1][2], node.u]
        if node.padding == "SAME":
            opname = 'convolution_2d_forward_SAME'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], inputsshape[1][0],
                             inputsshape[1][2], node.u]
    if node.name == "Convolution2DBackward":
        if node.padding == "VALID":
            if node.type == 0:
                opname = 'convolution_backward_data_2d_VALID'
                inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], inputsshape[1][0],
                                 inputsshape[1][2], node.u]
            if node.type == 1:
                opname = 'convolution_backward_filter_2d_VALID'
                inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], inputsshape[1][0],
                                 inputsshape[1][2], node.u]
        if node.padding == "SAME":
            if node.type == 0:
                opname = 'convolution_backward_data_2d_SAME'
                inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], inputsshape[1][0],
                                 inputsshape[1][2], node.u]
            if node.type == 1:
                opname = 'convolution_backward_filter_2d_SAME'
                inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], inputsshape[1][0],
                                 inputsshape[1][2], node.u]
    if node.name == "DropoutForward":
        opname = 'dropout_forward'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], opname.dropout]
    if node.name == "FullyDropoutForward":
        opname = 'dropout_forward'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], 1, opname.dropout]
    if node.name == "DropoutBackward":
        opname = 'dropout_backward'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], opname.dropout]
    if node.name == "FullyDropoutBackward":
        opname = 'dropout_backward'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], 1, opname.dropout]
    if node.name == "BNForward":
        opname = 'bn_forward_pre_activation'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2]]
    if node.name == "BNBackward":
        opname = 'bn_backward_pre_activation'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2]]
    if node.name == "Pooling2DForward":
        if node.poolingMode == "max":
            opname = 'pooling_2d_forward_max'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], node.filter_h, node.u]
        if node.poolingMode == "mean":
            opname = 'pooling_2d_forward_mean'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], node.filter_h, node.u]
    if node.name == "Pooling2DBackward":
        if node.inputs[2].poolingMode == "max":
            opname = 'pooling_2d_backward_max'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], node.inputs[2].filter_h, node.inputs[2].u]
        if node.inputs[2].poolingMode == "mean":
            opname = 'pooling_2d_backward_mean'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], node.inputs[2].filter_h, node.inputs[2].u]
    if node.name == "BroadcastTo":
        if node.type == "NHWC":
            opname = 'broadcast_to_NHWC'
            inputsofmodel = [inputsshape[1][0], inputsshape[1][1]]
        if node.type == "NCHW":
            opname = 'broadcast_to_NCHW'
            inputsofmodel = [inputsshape[1][0], inputsshape[1][1], inputsshape[1][2]]
    if node.name == "BroadcastToGradient":
        if node.type == "NHWC":
            opname = 'reduce_sum_new_NHWC'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
        if node.type == "NCHW":
            opname = 'reduce_sum_new_NCHW'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2]]
    if node.name == "CrossOp":
        opname = 'cross'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
    if node.name == "CrossBackwardOp":
        opname = 'cross_backward'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
    #这个node的计算用了两个gpuop
    if node.name == "AdamOp":
        opname0 = 'adam_mv'
        inputsofmodel0 = [inputsshape[0][1], inputsshape[0][0], inputsshape[0][2]]
        opname1 = 'adam_compute'
        return opname0,opname1,inputsofmodel0
    if node.name == "ActivationForward":
        if node.activationMode == "relu":
            opname = 'activation_forward_relu'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2]]
        if node.activationMode == "softmax":
            opname = 'activation_forward_softmax'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2]]
    if node.name == "FullyActivationForward":
        if node.activationMode == "relu":
            opname = 'activation_forward_relu'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], 1]
        if node.activationMode == "softmax":
            opname = 'activation_forward_softmax'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], 1]
    if node.name == "ActivationBackward":
        if node.activationMode == "relu":
            opname = 'activation_backward_relu'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2]]
        if node.activationMode == "softmax":
            opname = 'activation_backward_softmax'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2]]
    if node.name == "FullyActivationBackward":
        if node.activationMode == "relu":
            opname = 'activation_backward_relu'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], 1]
        if node.activationMode == "softmax":
            opname = 'activation_backward_softmax'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], 1]
    if node.name == "MatMul":
        opname = 'matrix_multiply'
        n1 = inputsshape[0][0]
        n2 = inputsshape[0][1]
        n3 = inputsshape[1][1]
        if node.matmul_attr_trans_A == True:
            n1 = inputsshape[0][1]
            n2 = inputsshape[0][0]
        if node.matmul_attr_trans_B == True:
            n3 = inputsshape[1][0]
        inputsofmodel = [n1, n2, n3]
    if node.name == "+":
        opname = 'matrix_elementwise_add'
        n = inputsshape[0][0]
        m = inputsshape[0][1]
        if len(inputsshape[0]) == 4:
            n = inputsshape[0][0] * inputsshape[0][1]
            m = inputsshape[0][2] * inputsshape[0][3]
        inputsofmodel = [n, m]
    if node.name == "ConcatForward":
        opname = 'concat_forward'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
    if node.name == "ConcatBackward":
        if node.type == 0:
            opname = 'concat_a_backward'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
        if node.type == 1:
            opname = 'concat_b_backward'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
    if node.name == "SgdOp":
        opname = 'sgd_update'
        inputsofmodel = [inputsshape[0][1], inputsshape[0][0], inputsshape[0][2]]
    if node.name == "*" or node.name == "Flatten" or node.name == "FlattenGradient":
        opname = 'matrix_elementwise_multiply_by_const'
        n = inputsshape[0][0]
        m = inputsshape[0][1]
        if len(inputsshape[0]) == 4:
            n = inputsshape[0][0] * inputsshape[0][1]
            m = inputsshape[0][2] * inputsshape[0][3]
        inputsofmodel = [m, n]
    if node.name == "Zeroslike" or node.name == "Oneslike":
        opname = 'array_set'
        n = inputsshape[0][0]
        m = inputsshape[0][1]
        if len(inputsshape[0]) == 4:
            n = inputsshape[0][0] * inputsshape[0][1]
            m = inputsshape[0][2] * inputsshape[0][3]
        inputsofmodel = [m, n]
    return opname,inputsofmodel


