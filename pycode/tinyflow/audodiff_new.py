""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import

import threading
import time
import numpy as np
from . import ndarray, gpu_op
import random
import queue
import datetime

index_to_cpu_map = {}
index_to_cpu_flag = {}
index_to_gpu_map = {}


class MemoryManagerController(threading.Thread):
    def __init__(self, control_queue: queue.Queue, have_done_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.will_do_queue = queue.Queue()
        self.have_done_queue = have_done_queue
        self.control_queue = control_queue
        # todo hard code with device id again, may need to change
        self.cpu_ctx = ndarray.cpu(0)
        self.gpu_ctx = ndarray.gpu(0)
        self.memoryManager = MemoryManager(self.will_do_queue, self.have_done_queue)
        self.memoryManager.start()

    def run(self):
        while True:
            # todo 接口内容：wait_time: 距离上一次swap的间隔时间，node_index和node_ndarray同Manager中的定义
            # todo 在此处检查当前移动是否需要，即检查是否已经在对应的ctx中，加入变量move_to_gpu
            # (wait_time, node_index, node_ndarray, move_to_gpu)
            control_message = self.control_queue.get(block=True)
            wait_time = control_message[0]
            node_index = control_message[1]
            move_to_gpu = control_message[2]
            # print(node_index, move_to_gpu)
            time.sleep(wait_time / 1000.0)
            self.will_do_queue.put((node_index, move_to_gpu))


class MemoryManager(threading.Thread):
    def __init__(self, will_do_queue: queue.Queue, have_done_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.will_do_queue = will_do_queue
        self.have_done_queue = have_done_queue
        # todo hard code with device id again, may need to change
        self.cpu_ctx = ndarray.cpu(0)
        self.gpu_ctx = ndarray.gpu(0)
        self.cudaSwapStream = gpu_op.create_cudaStream()

    def run(self):
        while (True):
            node = self.will_do_queue.get(block=True)
            node_index = node[0]
            move_to_gpu = node[1]
            node_ndarray_new = None


            global index_to_cpu_map
            global index_to_gpu_map


            if move_to_gpu == 0:
                node_ndarray = index_to_gpu_map[node_index]
                node_ndarray.copyto(index_to_cpu_map[node_index], self.cudaSwapStream)
                index_to_cpu_flag[node_index] = True
                index_to_gpu_map[node_index] = None
                # print("swap finish: node " + str(node_index) + " to " + str(move_to_gpu))

            else:
                node_ndarray = index_to_cpu_map[node_index]
                # time1 = datetime.datetime.now()

                node_ndarray_new = ndarray.empty(node_ndarray.shape, self.gpu_ctx)
                # time2 = datetime.datetime.now()

                node_ndarray.copyto(node_ndarray_new, self.cudaSwapStream)
                if index_to_gpu_map[node_index] is None:
                    index_to_gpu_map[node_index] = node_ndarray_new
                else:
                    print("swap in 和 passive import 重合")
                # print("swap finish: node " + str(node_index) + " to " + str(move_to_gpu))
                # print((time2 - time1).microseconds)




class Node(object):
    """Node in a computation graph."""

    def __init__(self):
        """Constructor, new node is indirectly created by Op object call method.

            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object,
                e.g. add_op if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant.
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""
        self.index = 0
        self.array_status = 0
        # todo array_status 0 - cpu ,  1 - gpu , 2 - trans from cpu to gpu, 3 - trans from gpu to cpu
        self.control_message_in = []
        self.control_message_in_time = 0
        self.control_message_out = []
        self.control_message_out_time = 0
        self.recompute_list = []
        self.release_list = []
        self.runtime = 0.01

        # 是不是参数
        self.issgd = 0
        self.isw = 0

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __mul__(self, other):
        """Multiplying two nodes return a new node."""
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            # Mul by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = mul_byconst_op(self, other)
        return new_node

        # Allow left-hand-side add and multiply.

    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return self.name


# 参数用
def Variable(name):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    placeholder_node.isw = 1
    return placeholder_node

    # 数据用


def Placeholder(name):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    placeholder_node.isw = 0
    return placeholder_node


class Op(object):
    """Op represents operations performed on nodes."""

    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.
        output_val: output value of the node, modified in-place.
        use_numpy: bool flag whether to use numpy for compute
        """
        raise NotImplementedError

        return 0

    def gradient(self, node, output_grad):
        """Given output gradient, compute partial gradient to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        """Given shapes of input nodes, compute shape of output node.

        Implementation note:
        It's simpler to treat shape of constants as (1,), so that constants can
        be stored as a numpy array too and you would need fewer special case
        handling.

        Parameters
        ----------
        node: node whose shape is being inferred.
        input_shapes: shapes of input nodes.

        Returns
        -------
        A tuple representing the shape of output node.
        """
        raise NotImplementedError


class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 2

        if use_numpy:
            # output_val[:] allows modify in-place
            output_val[:] = input_vals[0] + input_vals[1]
        else:
            if input_vals[0].shape == input_vals[1].shape:
                gpu_op.matrix_elementwise_add(
                    input_vals[0], input_vals[1], output_val, cudaStream)
            else:
                if input_vals[1].shape == (1,):
                    const_val = input_vals[1].asnumpy()[0]
                    gpu_op.matrix_elementwise_add_by_const(
                        input_vals[0], const_val, output_val, cudaStream)
                elif input_vals[0].shape == (1,):
                    const_val = input_vals[0].asnumpy()[0]
                    gpu_op.matrix_elementwise_add_by_const(
                        input_vals[1], const_val, output_val, cudaStream)

        return 0

    def gradient(self, node, output_grad):
        return [output_grad, output_grad]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""

        def get_ele_num(shape):
            num = 1
            for size_ in shape:
                num *= size_
            return num

        if input_shapes[0] == input_shapes[1]:
            return input_shapes[0]
        elif input_shapes[0] == (1,) or get_ele_num(input_shapes[0]):
            return input_shapes[1]
        elif input_shapes[1] == (1,) or get_ele_num(input_shapes[1]):
            return input_shapes[0]


class AddByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = input_vals[0] + node.const_attr
        else:
            gpu_op.matrix_elementwise_add_by_const(
                input_vals[0], node.const_attr, output_val, cudaStream)

        return 0

    def gradient(self, node, output_grad):
        return [output_grad]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = input_vals[0] * input_vals[1]
        else:
            if input_vals[0].shape == input_vals[1].shape:
                gpu_op.matrix_elementwise_multiply(
                    input_vals[0], input_vals[1], output_val, cudaStream)
            else:
                if input_vals[1].shape == (1,):
                    const_val = input_vals[1].asnumpy()[0]
                    gpu_op.matrix_elementwise_multiply_by_const(
                        input_vals[0], const_val, output_val, cudaStream)
                elif input_vals[0].shape == (1,):
                    const_val = input_vals[0].asnumpy()[0]
                    gpu_op.matrix_elementwise_multiply_by_const(
                        input_vals[1], const_val, output_val, cudaStream)

        return 0

    def gradient(self, node, output_grad):
        return [node.inputs[1] * output_grad, node.inputs[0] * output_grad]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        if input_shapes[0] == input_shapes[1]:
            return input_shapes[0]
        elif input_shapes[0] == (1,):
            return input_shapes[1]
        elif input_shapes[1] == (1,):
            return input_shapes[0]


class MulByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = input_vals[0] * node.const_attr
        else:
            gpu_op.matrix_elementwise_multiply_by_const(
                input_vals[0], node.const_attr, output_val, cudaStream)

        return 0

    def gradient(self, node, output_grad):
        return [node.const_attr * output_grad]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (
            node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):

        if use_numpy:
            if ((node.matmul_attr_trans_A is False) and
                    (node.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(input_vals[0], input_vals[1])
            elif ((node.matmul_attr_trans_A is True) and
                  (node.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0]), input_vals[1])
            elif ((node.matmul_attr_trans_A is False) and
                  (node.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    input_vals[0], np.transpose(input_vals[1]))
            elif ((node.matmul_attr_trans_A is True) and
                  (node.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0]), np.transpose(input_vals[1]))
        else:

            gpu_op.matrix_multiply(
                input_vals[0], node.matmul_attr_trans_A,
                input_vals[1], node.matmul_attr_trans_B,
                output_val, cublasHandle, cudaStream)

        return 0

    def gradient(self, node, output_grad):
        if ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is False)):
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            lhs_grad = matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                node.inputs[0], output_grad, trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
              (node.matmul_attr_trans_B is False)):
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A dY
            lhs_grad = matmul_op(
                node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                node.inputs[0], output_grad, trans_A=False, trans_B=False)
        elif ((node.matmul_attr_trans_A is False) and
              (node.matmul_attr_trans_B is True)):
            # if Y=A B^T, then dA=dY B, dB=(A^T dY)^T=dY^T A
            lhs_grad = matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=False)
            rhs_grad = matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
              (node.matmul_attr_trans_B is True)):
            # if Y=A^T B^T, then dA=(dY B^T)^T=B^T dY^T, dB=(A^T dY)^T=dY^T A^T
            lhs_grad = matmul_op(
                node.inputs[1], output_grad, trans_A=True, trans_B=True)
            rhs_grad = matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=True)
        return [lhs_grad, rhs_grad]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        if node.matmul_attr_trans_A is False and node.matmul_attr_trans_B is False:
            return input_shapes[0][0], input_shapes[1][1]
        elif node.matmul_attr_trans_A is False and node.matmul_attr_trans_B is True:
            return input_shapes[0][0], input_shapes[1][0]
        elif node.matmul_attr_trans_A is True and node.matmul_attr_trans_B is False:
            return input_shapes[0][1], input_shapes[1][1]
        else:
            return input_shapes[0][1], input_shapes[1][0]


class PlaceholderOp(Op):
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert False, "placeholder %s values provided by feed_dict" % node.name

        return 0

    def gradient(self, node, output_grad):
        return None

    def infer_shape(self, node, input_shapes, cudnnHandle):
        assert False, "placeholder %s shape provided by feed_shape" % node.name


class ZerosLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.zeros(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.zeros(input_vals[0].shape)
        else:
            gpu_op.array_set(output_val, 0, cudaStream)

        return 0

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        """If input_shape is a vector, simpler to return (1,)"""
        return input_shapes[0]


class OnesLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.ones(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.ones(input_vals[0].shape)
        else:
            gpu_op.array_set(output_val, 1, cudaStream)

        return 0

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        """If input_shape is a vector, simpler to return (1,)"""
        return input_shapes[0]


class ReduceSumAxisZeroOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.sum(node_A, axis=0).
        Only support common-case axis=0 reduction for simplicity of gradient.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ReduceSumAxisZero(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 1
        if use_numpy:
            assert (isinstance(input_vals[0], np.ndarray))
            output_val[:] = np.sum(input_vals[0], axis=0)
        else:
            gpu_op.reduce_sum_axis_zero(input_vals[0], output_val)

        return 0

    def gradient(self, node, output_grad):
        return [broadcastto_op(output_grad, node.inputs[0])]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        """summation reduction axis = 0
        e.g. (3,4,5)->(4,5)
        for vector, simpler to do (3,)->(1,)
        """
        assert len(input_shapes) == 1
        if len(input_shapes[0]) == 1:
            return (1,)
        return input_shapes[0][1:]


class BroadcastToOp(Op):
    def __call__(self, node_A, node_B, type="NHWC"):
        """Creates a node that represents np.broadcast_to(node_A, node_B.shape).
        .e.g. (3,4)->(2,5,3,4) to make gradient simple.
        e.g(1)->(6)
        (6)->(3,6,3,4)
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "BroadcastTo(%s,%s.shape)" % (node_A.name, node_B.name)
        new_node.type = type
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        gpu_op.broadcast_to(input_vals[0], output_val, node.type, cudaStream)

        return 0

    def gradient(self, node, output_grad):
        grad_A = broadcastto_gradient_op(output_grad, node.inputs[0], node.type)
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[1]


class BroadcastToGradientOp(Op):
    def __call__(self, node_A, node_B, type):
        # A: outgrad (2,5,3,4), B: (3,4)
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "BroadcastToGradient(%s,%s.shape)" % (node_A.name, node_B.name)
        new_node.type = type
        new_node.cudnnlist = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        # gpu_op.broadcast_to_backward(input_vals[0], output_val, node.type)

        # tic = time.time()
        memorytoSaving = gpu_op.reduce_sum_new(input_vals[0], output_val, node.cudnnlist[0], cudnnHandle, cudaStream)

        return memorytoSaving

        # toc = time.time()
        # print("use time1: " + str(toc - tic))

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        inputshape, outputshape = gpu_op.reduce_sum_get_real_shape(input_shapes[0], input_shapes[1], node.type)

        node.cudnnlist[0] = gpu_op.reduce_sum_get_cudnnlist(inputshape, outputshape, node.type, cudnnHandle)

        return input_shapes[1]


def softmax_func(y):
    """Numerically stable softmax."""
    b = y - np.max(y, axis=1, keepdims=True)
    expb = np.exp(b)
    softmax = expb / np.sum(expb, axis=1, keepdims=True)
    return softmax


class SoftmaxCrossEntropyOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "SoftmaxXEntropy(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 2
        y = input_vals[0]
        y_ = input_vals[1]
        if use_numpy:
            softmax = softmax_func(y)
            cross_entropy = np.mean(
                -np.sum(y_ * np.log(softmax), axis=1), keepdims=True)
            output_val[:] = cross_entropy
        else:
            gpu_op.softmax_cross_entropy(y, y_, output_val, cudaStream)

        return 0

    def gradient(self, node, output_grad):
        grad_A = (softmax_op(node.inputs[0]) + -1 * node.inputs[1]) * output_grad
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return (1,)


class SoftmaxOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Softmax(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = softmax_func(input_vals[0])
        else:
            gpu_op.softmax(input_vals[0], output_val, cudaStream)

        return 0

    def gradient(self, node, output_grad):
        # Do not directly use SoftmaxOp, use SoftmaxCrossEntropyOp instead.
        # Not allowing taking 2nd derivative of SoftmaxCrossEntropyOp.
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Relu(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.maximum(input_vals[0], 0)
        else:
            gpu_op.relu(input_vals[0], output_val)

        return 0

    def gradient(self, node, output_grad):
        return [relu_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class ReluGradientOp(Op):
    def __call__(self, node_A, node_B):
        """node_B is output_grad"""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ReluGradient(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 2
        if use_numpy:
            # heaviside function, 0.5 at x=0
            output_val[:] = (np.sign(input_vals[0]) + 1) * 0.5 * input_vals[1]
        else:
            gpu_op.relu_gradient(input_vals[0], input_vals[1], output_val)

        return 0

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class ExpOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Exp(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert len(input_vals) == 1

        gpu_op.matrix_exp(input_vals[0], output_val)

        return 0

    def gradient(self, node, output_grad):
        return [output_grad * node]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class LogOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Log(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert len(input_vals) == 1

        gpu_op.matrix_log(input_vals[0], output_val)

        return 0

    def gradient(self, node, output_grad):
        return [output_grad * reverse_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class ReverseOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Reverse(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert len(input_vals) == 1

        gpu_op.matrix_reverse(input_vals[0], output_val)

        return 0

    def gradient(self, node, output_grad):
        return [output_grad * (-1) * pow_op(node, 2)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class PowOp(Op):
    def __call__(self, node_A, val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Pow(%s)" % (node_A.name)
        new_node.val = val
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert len(input_vals) == 1

        gpu_op.matrix_pow(input_vals[0], node.val, output_val)

        return 0

    def gradient(self, node, output_grad):
        return [node.val * output_grad * pow_op(node.inputs[0], node.val - 1)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class Convolution1DForwardOp(Op):
    def __call__(self, node_A, node_B, dataformat, padding, v):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "Convolution1DForward(%s)withfilter(%s)" % (node_A.name, node_B.name)
        new_node.dataformat = dataformat
        new_node.padding = padding
        new_node.v = v
        new_node.cudnnlist = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 2

        memorytoSaving = gpu_op.convolution_1d_forward(input_vals[0], input_vals[1], output_val, node.cudnnlist[0],
                                                       cudnnHandle)
        return memorytoSaving

    def gradient(self, node, output_grad):
        return [convolution_1d_backward_op(node.inputs[0], node.inputs[1], output_grad, 0, node.cudnnlist),
                convolution_1d_backward_op(node.inputs[0], node.inputs[1], output_grad, 1, node.cudnnlist)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        outshapes, node.cudnnlist[0] = gpu_op.convolution_1d_forward_get_out_shape(input_shapes[0], input_shapes[1],
                                                                                   node.dataformat, node.padding,
                                                                                   node.v)
        return outshapes


class Convolution1DBackwardOp(Op):
    def __call__(self, node_A, node_B, node_C, type, cudnnlist):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_C]
        new_node.name = "Convolution1DBackward(%s)withfilter(%s)withdoutput(%s)withtype(%s)" % (
            node_A.name, node_B.name, node_C.name, type)
        # 0 is dinput, 1 is dfilter
        new_node.type = type
        new_node.cudnnlist = cudnnlist
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 3
        assert isinstance(input_vals[0], ndarray.NDArray)
        assert isinstance(input_vals[1], ndarray.NDArray)
        assert isinstance(input_vals[2], ndarray.NDArray)

        if node.type == 0:
            memorytoSaving = gpu_op.convolution_backward_data(input_vals[0], input_vals[2], input_vals[1], output_val,
                                                              node.cudnnlist[0], cudnnHandle, cudaStream)
        if node.type == 1:
            memorytoSaving = gpu_op.convolution_backward_filter(input_vals[0], input_vals[2], input_vals[1], output_val,
                                                                node.cudnnlist[0], cudnnHandle, cudaStream)
        return memorytoSaving

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        if node.type == 0:
            return input_shapes[0]
        else:
            return input_shapes[1]


class Convolution2DForwardOp(Op):
    def __call__(self, node_A, node_B, dataformat, padding, u, v):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "Convolution2DForward(%s)withfilter(%s)" % (node_A.name, node_B.name)
        new_node.dataformat = dataformat
        new_node.padding = padding
        new_node.u = u
        new_node.v = v
        new_node.cudnnlist = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 2

        memorytoSaving = gpu_op.convolution_2d_forward(input_vals[0], input_vals[1], output_val, node.cudnnlist[0],
                                                       cudnnHandle, cudaStream)

        return memorytoSaving

    def gradient(self, node, output_grad):
        return [convolution_2d_backward_op(node.inputs[0], node.inputs[1], output_grad, 0, node.cudnnlist),
                convolution_2d_backward_op(node.inputs[0], node.inputs[1], output_grad, 1, node.cudnnlist)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        outshapes, node.cudnnlist[0] = gpu_op.convolution_2d_forward_get_out_shape(input_shapes[0], input_shapes[1],
                                                                                   node.dataformat, node.padding,
                                                                                   node.u, node.v)

        return outshapes


class Convolution2DBackwardOp(Op):
    def __call__(self, node_A, node_B, node_C, type, cudnnlist):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_C]
        new_node.name = "Convolution2DBackward(%s)withfilter(%s)withdoutput(%s)withtype(%s)" % (
            node_A.name, node_B.name, node_C.name, type)
        # 0 is dinput, 1 is dfilter
        new_node.type = type

        new_node.cudnnlist = cudnnlist

        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 3
        assert isinstance(input_vals[0], ndarray.NDArray)
        assert isinstance(input_vals[1], ndarray.NDArray)
        assert isinstance(input_vals[2], ndarray.NDArray)

        if node.type == 0:
            memorytoSaving = gpu_op.convolution_backward_data(input_vals[0], input_vals[2], input_vals[1], output_val,
                                                              node.cudnnlist[0], cudnnHandle, cudaStream)
        if node.type == 1:
            memorytoSaving = gpu_op.convolution_backward_filter(input_vals[0], input_vals[2], input_vals[1], output_val,
                                                                node.cudnnlist[0], cudnnHandle, cudaStream)

        return memorytoSaving

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        if node.type == 0:
            return input_shapes[0]
        else:
            return input_shapes[1]


class Convolution3DForwardOp(Op):
    def __call__(self, node_A, node_B, dataformat, padding, s1, s2, s3):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "Convolution3DForward(%s)withfilter(%s)" % (node_A.name, node_B.name)
        new_node.dataformat = dataformat
        new_node.padding = padding
        new_node.s1 = s1
        new_node.s2 = s2
        new_node.s3 = s3
        new_node.cudnnlist = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 2

        memorytoSaving = gpu_op.convolution_3d_forward(input_vals[0], input_vals[1], output_val, node.cudnnlist[0],
                                                       cudnnHandle)
        return memorytoSaving

    def gradient(self, node, output_grad):
        return [convolution_3d_backward_op(node.inputs[0], node.inputs[1], output_grad, 0, node.cudnnlist),
                convolution_3d_backward_op(node.inputs[0], node.inputs[1], output_grad, 1, node.cudnnlist)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        outshapes, node.cudnnlist[0] = gpu_op.convolution_3d_forward_get_out_shape(input_shapes[0], input_shapes[1],
                                                                                   node.dataformat, node.padding,
                                                                                   node.s1, node.s2, node.s3)
        return outshapes


class Convolution3DBackwardOp(Op):
    def __call__(self, node_A, node_B, node_C, type, cache, cudnnlist):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_C]
        new_node.name = "Convolution3DBackward(%s)withfilter(%s)withdoutput(%s)withtype(%s)" % (
            node_A.name, node_B.name, node_C.name, type)
        # 0 is dinput, 1 is dfilter
        new_node.type = type
        new_node.cudnnlist = cudnnlist

        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 3
        assert isinstance(input_vals[0], ndarray.NDArray)
        assert isinstance(input_vals[1], ndarray.NDArray)
        assert isinstance(input_vals[2], ndarray.NDArray)
        if node.type == 0:
            memorytoSaving = gpu_op.convolution_backward_data(input_vals[0], input_vals[2], input_vals[1], output_val,
                                                              node.cudnnlist[0], cudnnHandle, cudaStream)
        if node.type == 1:
            memorytoSaving = gpu_op.convolution_backward_filter(input_vals[0], input_vals[2], input_vals[1], output_val,
                                                                node.cudnnlist[0], cudnnHandle, cudaStream)

        return memorytoSaving

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        if node.type == 0:
            return input_shapes[0]
        else:
            return input_shapes[1]


class FlattenOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Flatten(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        input_vals[0].copyto(output_val)
        return 0

    def gradient(self, node, output_grad):
        return [flatten_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        output_shape1 = 1
        output_shape0 = input_shapes[0][0]
        for i in range(1, len(input_shapes[0])):
            output_shape1 = output_shape1 * input_shapes[0][i]
        output_shape = (output_shape0, output_shape1)
        return output_shape


class FlattenGradientOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "FlattenGradient(%s)withdoutput(%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        input_vals[1].copyto(output_val)
        return 0

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class ActivationForwardOp(Op):
    def __call__(self, node_A, dataformat, activationMode):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ActivationForward(%s)" % (node_A.name)
        new_node.dataformat = dataformat
        new_node.activationMode = activationMode
        new_node.cudnnlist = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        gpu_op.activation_forward(input_vals[0], output_val, node.activationMode, node.cudnnlist[0], cudnnHandle,
                                  cudaStream)

        return 0

    def gradient(self, node, output_grad):
        return [activation_backward_op(node.inputs[0], output_grad, node, node.activationMode, node.cudnnlist)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        node.cudnnlist[0] = gpu_op.activation_get_cudnnlist(input_shapes[0], node.dataformat, node.activationMode)
        return input_shapes[0]


class ActivationBackwardOp(Op):
    def __call__(self, node_A, node_B, node_C, activationMode, cudnnlist):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_C]
        new_node.name = "ActivationBackwardwithinput(%s)withdoutput(%s)withoutput(%s)" % (
            node_A.name, node_B.name, node_C.name)
        new_node.activationMode = activationMode
        new_node.cudnnlist = cudnnlist
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False

        gpu_op.activation_backward(input_vals[0], output_val, input_vals[2], input_vals[1], node.activationMode,
                                   node.cudnnlist[0], cudnnHandle, cudaStream)
        return 0

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class Pooling1DForwardOp(Op):
    def __call__(self, node_A, dataformat, poolingMode, pad_w, v, filter_w):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Pooling1DForward(%s)" % (node_A.name)
        new_node.dataformat = dataformat
        new_node.poolingMode = poolingMode
        new_node.pad_w = pad_w
        new_node.v = v
        new_node.filter_w = filter_w
        new_node.cudnnlist = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert isinstance(input_vals[0], ndarray.NDArray)
        gpu_op.pooling_1d_forward(input_vals[0], output_val, node.cudnnlist[0], cudnnHandle)

        return 0

    def gradient(self, node, output_grad):
        return [pooling_1d_backward_op(node.inputs[0], output_grad, node, node.cudnnlist)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        outshapes, node.cudnnlist[0] = gpu_op.pooling_1d_forward_get_out_shape(input_shapes[0], node.dataformat,
                                                                               node.poolingMode, node.pad_w, node.v,
                                                                               node.filter_w)
        return outshapes


class Pooling1DBackwardOp(Op):
    def __call__(self, node_A, node_B, node_C, cudnnlist):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_C]
        new_node.name = "Pooling1DBackwardwithinput(%s)withdoutput(%s)withoutput(%s)" % (
            node_A.name, node_B.name, node_C.name)
        new_node.cudnnlist = cudnnlist
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False

        gpu_op.pooling_1d_backward(input_vals[0], input_vals[2], input_vals[1], output_val, node.cudnnlist[0],
                                   cudnnHandle)
        return 0

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class Pooling2DForwardOp(Op):
    def __call__(self, node_A, dataformat, poolingMode, pad_h, pad_w, u, v, filter_h, filter_w):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Pooling2DForward(%s)" % (node_A.name)
        new_node.dataformat = dataformat
        new_node.poolingMode = poolingMode
        new_node.pad_h = pad_h
        new_node.pad_w = pad_w
        new_node.u = u
        new_node.v = v
        new_node.filter_h = filter_h
        new_node.filter_w = filter_w
        new_node.cudnnlist = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert isinstance(input_vals[0], ndarray.NDArray)

        gpu_op.pooling_2d_forward(input_vals[0], output_val, node.cudnnlist[0], cudnnHandle, cudaStream)

        return 0

    def gradient(self, node, output_grad):
        return [pooling_2d_backward_op(node.inputs[0], output_grad, node, node.cudnnlist)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        outshapes, node.cudnnlist[0] = gpu_op.pooling_2d_forward_get_out_shape(input_shapes[0], node.dataformat,
                                                                               node.poolingMode, node.pad_h, node.pad_w,
                                                                               node.u, node.v, node.filter_h,
                                                                               node.filter_w)
        return outshapes


class Pooling2DBackwardOp(Op):
    def __call__(self, node_A, node_B, node_C, cudnnlist):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_C]
        new_node.name = "Pooling2DBackwardwithinput(%s)withdoutput(%s)withoutput(%s)" % (
            node_A.name, node_B.name, node_C.name)
        new_node.cudnnlist = cudnnlist
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        gpu_op.pooling_2d_backward(input_vals[0], input_vals[2], input_vals[1], output_val, node.cudnnlist[0],
                                   cudnnHandle, cudaStream)
        return 0

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class Pooling3DForwardOp(Op):
    def __call__(self, node_A, dataformat, poolingMode, pad1, pad2, pad3, s1, s2, s3, filter1, filter2, filter3):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Pooling3DForward(%s)" % (node_A.name)
        new_node.dataformat = dataformat
        new_node.poolingMode = poolingMode
        new_node.pad1 = pad1
        new_node.pad2 = pad2
        new_node.pad3 = pad3
        new_node.s1 = s1
        new_node.s2 = s2
        new_node.s3 = s3
        new_node.filter1 = filter1
        new_node.filter2 = filter2
        new_node.filter3 = filter3
        new_node.cudnnlist = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert isinstance(input_vals[0], ndarray.NDArray)

        gpu_op.pooling_3d_forward(input_vals[0], output_val, node.cudnnlist[0], cudnnHandle)
        return 0

    def gradient(self, node, output_grad):
        return [pooling_3d_backward_op(node.inputs[0], output_grad, node, node.cudnnlist)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        outshapes, node.cudnnlist[0] = gpu_op.pooling_3d_forward_get_out_shape(input_shapes[0], node.dataformat,
                                                                               node.poolingMode, node.pad1, node.pad2,
                                                                               node.pad3, node.s1, node.s2, node.s3,
                                                                               node.filter1, node.filter2, node.filter3)
        return outshapes


class Pooling3DBackwardOp(Op):
    def __call__(self, node_A, node_B, node_C, cudnnlist):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_C]
        new_node.name = "Pooling3DBackwardwithinput(%s)withdoutput(%s)withoutput(%s)" % (
            node_A.name, node_B.name, node_C.name)
        new_node.cudnnlist = cudnnlist
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        gpu_op.pooling_3d_backward(input_vals[0], input_vals[2], input_vals[1], output_val, node.cudnnlist[0],
                                   cudnnHandle)
        return 0

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class DropoutForwardOp(Op):
    def __call__(self, node_A, dataformat, dropout):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "DropoutForward(%s)" % (node_A.name)
        new_node.dataformat = dataformat
        new_node.dropout = dropout
        new_node.seed = [0]
        new_node.reserveSpace_p = [0]
        new_node.cudnnlist = [0]
        new_node.inputd = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert isinstance(input_vals[0], ndarray.NDArray)
        node.seed[0] = random.randint(0, 100)
        node.reserveSpace_p[0], node.cudnnlist[0], memorytoSaving, err = gpu_op.dropout_forward(input_vals[0], output_val,
                                                                                           node.dataformat,
                                                                                           node.dropout, node.seed[0],
                                                                                           node.inputd[0], cudnnHandle,
                                                                                           cudaStream)
        return memorytoSaving

    def gradient(self, node, output_grad):
        return [dropout_backward_op(output_grad, node.reserveSpace_p, node.cudnnlist)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        node.inputd[0] = gpu_op.get_input_descriptor(input_shapes[0], node.dataformat)
        return input_shapes[0]


class DropoutBackwardOp(Op):
    def __call__(self, node_A, reserveSpace_p, cudnnlist):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "DropoutBackwardwithdoutput(%s)" % (node_A.name)
        new_node.reserveSpace_p = reserveSpace_p
        new_node.cudnnlist = cudnnlist
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        gpu_op.dropout_backward(input_vals[0], output_val, node.reserveSpace_p[0], node.cudnnlist[0], cudnnHandle,
                                cudaStream)
        return 0

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class FullyDropoutForwardOp(Op):
    def __call__(self, node_A, dataformat, dropout):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "FullyDropoutForward(%s)" % (node_A.name)
        new_node.dataformat = dataformat
        new_node.dropout = dropout
        new_node.seed = [0]
        new_node.reserveSpace_p = [0]
        new_node.cudnnlist = [0]
        new_node.inputd = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert isinstance(input_vals[0], ndarray.NDArray)
        node.seed[0] = random.randint(0, 100)
        input = input_vals[0]
        # inputs = input.reshape((input.shape[0], 1, input.shape[1]))

        node.reserveSpace_p[0], node.cudnnlist[0], memorytoSaving, err = gpu_op.dropout_forward(input, output_val,
                                                                                           node.dataformat,
                                                                                           node.dropout, node.seed[0],
                                                                                           node.inputd[0], cudnnHandle,
                                                                                           cudaStream)
        return memorytoSaving

    def gradient(self, node, output_grad):
        return [fullydropout_backward_op(output_grad, node.reserveSpace_p, node.cudnnlist)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        newinputshapes = (input_shapes[0][0], 1, input_shapes[0][1])
        node.inputd[0] = gpu_op.get_input_descriptor(newinputshapes, node.dataformat)
        return input_shapes[0]


class FullyDropoutBackwardOp(Op):
    def __call__(self, node_A, reserveSpace_p, cudnnlist):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "FullyDropoutBackwardwithdoutput(%s)" % (node_A.name)
        new_node.reserveSpace_p = reserveSpace_p
        new_node.cudnnlist = cudnnlist
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False

        gpu_op.dropout_backward(input_vals[0], output_val, node.reserveSpace_p[0], node.cudnnlist[0], cudnnHandle,
                                cudaStream)
        return 0

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class FullyActivationForwardOp(Op):
    def __call__(self, node_A, dataformat, activationMode):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "FullyActivationForward(%s)" % (node_A.name)
        new_node.dataformat = dataformat
        new_node.activationMode = activationMode
        new_node.cudnnlist = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        # print("fullyactivation_start")
        assert use_numpy == False

        gpu_op.activation_forward(input_vals[0], output_val, node.activationMode, node.cudnnlist[0], cudnnHandle,
                                  cudaStream)

        # print("fullyactivation_end")
        return 0

    def gradient(self, node, output_grad):
        return [
            fullyactivation_backward_op(node.inputs[0], output_grad, node, node.activationMode, node.cudnnlist)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        newinputshapes = (input_shapes[0][0], 1, input_shapes[0][1])
        node.cudnnlist[0] = gpu_op.activation_get_cudnnlist(newinputshapes, node.dataformat, node.activationMode)
        return input_shapes[0]


class FullyActivationBackwardOp(Op):
    def __call__(self, node_A, node_B, node_C, activationMode, cudnnlist):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_C]
        new_node.name = "FullyActivationBackwardwithinput(%s)withdoutput(%s)withoutput(%s)" % (
            node_A.name, node_B.name, node_C.name)
        new_node.activationMode = activationMode
        new_node.cudnnlist = cudnnlist
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        # print("FullyActivationBackwardOp_start")
        assert use_numpy == False

        gpu_op.activation_backward(input_vals[0], output_val, input_vals[2], input_vals[1], node.activationMode,
                                   node.cudnnlist[0], cudnnHandle, cudaStream)
        # print("FullyActivationBackwardOp_end")
        return 0

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class ReduceSumOp(Op):
    def __call__(self, node_A, axis=-1):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ReduceSum(%s)with(%s)" % (node_A.name, axis)
        new_node.axis = axis
        new_node.inputshape = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert len(input_vals) == 1
        gpu_op.reduce_sum(input_vals[0], output_val, node.axis)

        return 0

    def gradient(self, node, output_grad):
        return [reduce_sum_backward_op(output_grad, node.inputshape, node.axis)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        node.inputshape[0] = input_shapes[0]
        return gpu_op.reduce_sum_get_outshape(input_shapes[0], node.axis)


class ReduceSumBackwardOp(Op):
    def __call__(self, node_A, inputshape, axis):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ReduceSumBackward(%s)withinputshape(%s)withaxis(%s)" % (node_A.name, inputshape[0], axis)
        new_node.axis = axis
        new_node.inputshape = inputshape
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert len(input_vals) == 1
        gpu_op.reduce_sum_backward(input_vals[0], output_val, node.axis)

        return 0

    def gradient(self, node, output_grad):
        return NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return node.inputshape[0]


class ReduceMeanOp(Op):
    def __call__(self, node_A, axis=-1):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ReduceMean(%s)with(%s)" % (node_A.name, axis)
        new_node.axis = axis
        new_node.inputshape = [0, 0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert len(input_vals) == 1
        val = 1
        if node.axis == -1:
            val = gpu_op.get_shape_size(node.inputshape[0])
        else:
            val = node.inputshape[0][node.axis]
        node.inputshape[1] = val
        ctx = ndarray.gpu(0)
        outtmp = ndarray.empty(output_val.shape, ctx)
        gpu_op.reduce_sum(input_vals[0], outtmp, node.axis)
        gpu_op.matrix_elementwise_multiply_by_const(outtmp, 1. / val, output_val, cudaStream)

        return 0

    def gradient(self, node, output_grad):
        return [reduce_mean_backward_op(output_grad, node.inputshape, node.axis)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        node.inputshape[0] = input_shapes[0]
        return gpu_op.reduce_sum_get_outshape(input_shapes[0], node.axis)


class ReduceMeanBackwardOp(Op):
    def __call__(self, node_A, inputshape, axis):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ReduceMeanBackward(%s)withinputshape(%s)withaxis(%s)" % (node_A.name, inputshape[0], axis)
        new_node.axis = axis
        new_node.inputshape = inputshape
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert len(input_vals) == 1
        gpu_op.reduce_sum_backward(input_vals[0], output_val, node.axis)
        outtmp = output_val.copyto(ndarray.gpu(0))
        val = 1. / node.inputshape[1]
        gpu_op.matrix_elementwise_multiply_by_const(outtmp, val, output_val, cudaStream)
        return 0

    def gradient(self, node, output_grad):
        return NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return node.inputshape[0]


class CrossEntropyOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "CrossEntropy(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, use_numpy=True):
        assert len(input_vals) == 2
        y = input_vals[0]
        y_ = input_vals[1]
        if use_numpy:
            cross_entropy = np.mean(
                -np.sum(y_ * np.log(y), axis=1), keepdims=True)
            output_val[:] = cross_entropy
        else:
            gpu_op.cross_entropy(y, y_, output_val)

        return 0

    def gradient(self, node, output_grad):
        re = node.inputs[0]
        r2 = reverse_op(node.inputs[0])
        reo = mul_byconst_op(re, -1)
        reo = add_byconst_op(reo, 1)
        reo = reverse_op(reo)
        re1 = node.inputs[1]
        y_ = mul_byconst_op(re1, -1)
        y_ = add_byconst_op(y_, 1)
        grad_A = (-1 * mul_op(node.inputs[1], r2) + mul_op(y_, reo)) * output_grad
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class L1lossOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "L1loss(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 2
        y = input_vals[0]
        y_ = input_vals[1]
        if use_numpy:
            L1 = np.mean(np.sum(np.abs(y_ - y), axis=1), keepdims=True)
            output_val[:] = L1
        else:
            gpu_op.L1loss(y, y_, output_val)

    def gradient(self, node, output_grad):
        return [l1loss_gradient_op(node.inputs[0], node.inputs[1], output_grad),
                zeroslike_op(node.inputs[1])]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return (1,)


class L1lossgradientOp(Op):
    def __call__(self, node_A, node_B, node_C):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_C]
        new_node.name = "L1lossgradient(%s,%s,%s)" % (node_A.name, node_B.name, node_C.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        gpu_op.l1loss_gradient(input_vals[0], input_vals[1], input_vals[2], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class L2lossOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "L2loss(%s,%s)" % (node_A.name, node_B.name)
        new_node.inputshape = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 2
        y = input_vals[0]
        y_ = input_vals[1]
        if use_numpy:
            L2 = np.mean(np.sum(np.square(y_ - y), axis=1), keepdims=True)
            output_val[:] = L2
        else:
            gpu_op.L2loss(y, y_, output_val)

    def gradient(self, node, output_grad):
        return [l2loss_gradient_op(node.inputs[0], node.inputs[1], output_grad),
                zeroslike_op(node.inputs[1])]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return (1,)


class L2lossgradientOp(Op):
    def __call__(self, node_A, node_B, node_C):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_C]
        new_node.name = "L2lossgradient(%s,%s,%s) " % (node_A.name, node_B.name, node_C.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        gpu_op.l2loss_gradient(input_vals[0], input_vals[1], input_vals[2], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class L1regularOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "L1regular(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 1
        y = input_vals[0]
        if use_numpy:
            L1 = np.mean(np.sum(np.abs(y), axis=1), keepdims=True)
            output_val[:] = L1
        else:
            gpu_op.L1regular(y, output_val)

    def gradient(self, node, output_grad):
        return [l1regular_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return (1,)


class L1regulargradientOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "L1regulargradient(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        gpu_op.l1regular_gradient(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class L2regularOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "L2regular(%s)" % (node_A.name)
        new_node.inputshape = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 1
        y = input_vals[0]
        if use_numpy:
            L2 = np.mean(np.sum(np.abs(y), axis=1), keepdims=True)
            output_val[:] = L2
        else:
            gpu_op.L2regular(y, output_val)

    def gradient(self, node, output_grad):
        return [l2regular_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return (1,)


class L2regulargradientOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "L2regulargradient(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        gpu_op.l2regular_gradient(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class BNForwardOp(Op):
    def __call__(self, node_A, dataformat, batchNormMode):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "BNForward(%s)" % (node_A.name)
        new_node.dataformat = dataformat
        new_node.batchNormMode = batchNormMode
        new_node.Save_p = [0, 0]
        new_node.n = 0
        new_node.cudnnlist = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert isinstance(input_vals[0], ndarray.NDArray)

        memorytoSaving = gpu_op.bn_forward(input_vals[0], output_val, node.batchNormMode, node.n, node.Save_p[0],
                                           node.Save_p[1], node.cudnnlist[0], cudnnHandle, cudaStream)
        node.n = node.n + 1

        return memorytoSaving

    def gradient(self, node, output_grad):
        return [bn_backward_op(node.inputs[0], output_grad, node.batchNormMode, node.Save_p, node.cudnnlist)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        node.Save_p[0], node.Save_p[1], node.cudnnlist[0] = gpu_op.bn_get_cudnnlist(input_shapes[0], node.dataformat,
                                                                                    node.batchNormMode)

        return input_shapes[0]


class BNBackwardOp(Op):
    def __call__(self, node_A, node_B, batchNormMode, Save_p, cudnnlist):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "BNBackward(%s)withdoutput(%s)" % (node_A.name, node_B.name)
        new_node.Save_p = Save_p
        new_node.batchNormMode = batchNormMode
        new_node.cudnnlist = cudnnlist
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False

        memorytoSaving = gpu_op.bn_backward(input_vals[0], input_vals[1], output_val, node.batchNormMode,
                                            node.Save_p[0], node.Save_p[1], node.cudnnlist[0], cudnnHandle, cudaStream)

        return memorytoSaving

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class FullyBNForwardOp(Op):
    def __call__(self, node_A, dataformat):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "FullyBNForward(%s)" % (node_A.name)
        new_node.dataformat = dataformat
        new_node.batchNormMode = "pre_activation"
        new_node.Save_p = [0, 0]
        new_node.n = 0
        new_node.cudnnlist = [0]
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert isinstance(input_vals[0], ndarray.NDArray)

        memorytoSaving = gpu_op.bn_forward(input_vals[0], output_val, node.batchNormMode, node.n, node.Save_p[0],
                                           node.Save_p[1], node.cudnnlist[0], cudnnHandle, cudaStream)

        node.n = node.n + 1
        return memorytoSaving

    def gradient(self, node, output_grad):
        return [fullybn_backward_op(node.inputs[0], output_grad, node.batchNormMode, node.Save_p, node.cudnnlist)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        newinputshapes = (input_shapes[0][0], 1, input_shapes[0][1])
        node.Save_p[0], node.Save_p[1], node.cudnnlist[0] = gpu_op.bn_get_cudnnlist(newinputshapes, node.dataformat,
                                                                                    node.batchNormMode)
        return input_shapes[0]


class FullyBNBackwardOp(Op):
    def __call__(self, node_A, node_B, batchNormMode, Save_p, cudnnlist):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "FullyBNBackward(%s)withdoutput(%s)" % (node_A.name, node_B.name)
        new_node.Save_p = Save_p
        new_node.batchNormMode = batchNormMode
        new_node.cudnnlist = cudnnlist
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        input = input_vals[0]
        # inputs = input.reshape((input.shape[0], 1, input.shape[1]))
        memorytoSaving = gpu_op.bn_backward(input, input_vals[1], output_val, node.batchNormMode, node.Save_p[0],
                                            node.Save_p[1], node.cudnnlist[0], cudnnHandle, cudaStream)
        return memorytoSaving

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class ConcatForwardOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ConcatForward(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        assert isinstance(input_vals[0], ndarray.NDArray)
        assert isinstance(input_vals[1], ndarray.NDArray)
        gpu_op.concat_forward(input_vals[0], input_vals[1], output_val, cudaStream)

    def gradient(self, node, output_grad):
        return [concat_backward_op(node.inputs[0], node.inputs[1], output_grad, 0),
                concat_backward_op(node.inputs[0], node.inputs[1], output_grad, 1)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        output_shape1 = input_shapes[0][1] + input_shapes[1][1]
        output_shape = list(input_shapes[0])
        output_shape[1] = output_shape1
        return tuple(output_shape)


class ConcatBackwardOp(Op):
    def __call__(self, node_A, node_B, node_C, type):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_C]
        new_node.name = "Concatbackward(%s,%s,%s)withtype(%s)" % (node_A.name, node_B.name, node_C.name, type)
        new_node.type = type
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert len(input_vals) == 3
        assert isinstance(input_vals[0], ndarray.NDArray)
        assert isinstance(input_vals[1], ndarray.NDArray)
        assert isinstance(input_vals[2], ndarray.NDArray)
        if node.type == 0:
            gpu_op.concat_a_backward(input_vals[0], input_vals[1], input_vals[2], output_val, cudaStream)
        if node.type == 1:
            gpu_op.concat_b_backward(input_vals[0], input_vals[1], input_vals[2], output_val, cudaStream)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        if node.type == 0:
            return input_shapes[0]
        else:
            return input_shapes[1]


class SqueezeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Squeeze(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        input_vals[0].copyto(output_val)
        return 0

    def gradient(self, node, output_grad):
        return [squeeze_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes, cudnnHandle):
        output_shape1 = input_shapes[0][1]
        output_shape0 = input_shapes[0][0]
        output_shape = (output_shape0, output_shape1)
        return output_shape


class SqueezeGradientOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "SqueezeGradient(%s)withdoutput(%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        input_vals[1].copyto(output_val)
        return 0

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


class SgdOp(Op):
    def __call__(self, node_A, node_B, learning_rate):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "SgdOp(%s,%s)" % (node_A.name, node_B.name)
        new_node.learning_rate = learning_rate
        return new_node

    def compute(self, node, input_vals, output_val, cudnnHandle, cublasHandle, cudaStream, use_numpy=False):
        assert use_numpy == False
        gpu_op.sgd_update(input_vals[0], input_vals[1], node.learning_rate, cudaStream)
        return 0

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes, cudnnHandle):
        return input_shapes[0]


def dense(X, W, b):
    xw = matmul_op(X, W)
    xwb = xw + broadcastto_op(b, xw)
    return xwb


def conv1withbias(input, filter, bias, dataformat, padding, stride):
    c = convolution_1d_forward_op(input, filter, dataformat, padding, stride)
    b = broadcastto_op(bias, c, dataformat)
    cb = c + b
    return cb


def conv2withbias(input, filter, bias, dataformat, padding, strideh, stridew):
    c = convolution_2d_forward_op(input, filter, dataformat, padding, strideh, stridew)
    b = broadcastto_op(bias, c, dataformat)
    cb = c + b
    return cb


def conv3withbias(input, filter, bias, dataformat, padding, stride1, stride2, stride3):
    c = convolution_3d_forward_op(input, filter, dataformat, padding, stride1, stride2, stride3)
    b = broadcastto_op(bias, c, dataformat)
    cb = c + b
    return cb

    # Create global singletons of operators.


add_op = AddOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
reducesumaxiszero_op = ReduceSumAxisZeroOp()
broadcastto_op = BroadcastToOp()
broadcastto_gradient_op = BroadcastToGradientOp()
softmaxcrossentropy_op = SoftmaxCrossEntropyOp()
softmax_op = SoftmaxOp()
relu_op = ReluOp()
relu_gradient_op = ReluGradientOp()
convolution_1d_forward_op = Convolution1DForwardOp()
convolution_1d_backward_op = Convolution1DBackwardOp()
convolution_2d_forward_op = Convolution2DForwardOp()
convolution_2d_backward_op = Convolution2DBackwardOp()
convolution_3d_forward_op = Convolution3DForwardOp()
convolution_3d_backward_op = Convolution3DBackwardOp()
flatten_op = FlattenOp()
flatten_gradient_op = FlattenGradientOp()
activation_forward_op = ActivationForwardOp()
activation_backward_op = ActivationBackwardOp()
pooling_1d_forward_op = Pooling1DForwardOp()
pooling_1d_backward_op = Pooling1DBackwardOp()
pooling_2d_forward_op = Pooling2DForwardOp()
pooling_2d_backward_op = Pooling2DBackwardOp()
pooling_3d_forward_op = Pooling3DForwardOp()
pooling_3d_backward_op = Pooling3DBackwardOp()
dropout_forward_op = DropoutForwardOp()
dropout_backward_op = DropoutBackwardOp()
fullydropout_forward_op = FullyDropoutForwardOp()
fullydropout_backward_op = FullyDropoutBackwardOp()
fullyactivation_forward_op = FullyActivationForwardOp()
fullyactivation_backward_op = FullyActivationBackwardOp()
exp_op = ExpOp()
log_op = LogOp()
reverse_op = ReverseOp()
pow_op = PowOp()
reduce_sum_op = ReduceSumOp()
reduce_sum_backward_op = ReduceSumBackwardOp()
reduce_mean_op = ReduceMeanOp()
reduce_mean_backward_op = ReduceMeanBackwardOp()
crossEntropy_op = CrossEntropyOp()  # 2分类用,没有求和
l1loss_op = L1lossOp()  #
l1loss_gradient_op = L1lossgradientOp()
l2loss_op = L2lossOp()  #
l2loss_gradient_op = L2lossgradientOp()
l1regular_op = L1regularOp()  #
l1regular_gradient_op = L1regulargradientOp()
l2regular_op = L2regularOp()  #
l2regular_gradient_op = L2regulargradientOp()
bn_forward_op = BNForwardOp()
bn_backward_op = BNBackwardOp()
fullybn_forward_op = FullyBNForwardOp()
fullybn_backward_op = FullyBNBackwardOp()
concat_forward_op = ConcatForwardOp()
concat_backward_op = ConcatBackwardOp()
squeeze_op = SqueezeOp()
squeeze_gradient_op = SqueezeGradientOp()
sgd_op = SgdOp()


# test use
def nodelist_to_name(nodelist):
    nodename = []
    for node in nodelist:
        nodename.append(node.name)
    return nodename


class Executor(object):
    """Executor computes values for given set of nodes in computation graph."""

    def __init__(self, eval_node_list, ctx, top_control_queue, top_message_queue):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        ctx: runtime DLContext, default is None which means np.ndarray on cpu
        topo_order: list of nodes in topological order
        node_to_shape_map: dict from node to shape of the node
        node_to_arr_map: dict from node to ndarray.NDArray allocated for node
        feed_shapes: shapes of feed_dict from last run(...)
        """
        self.eval_node_list = eval_node_list
        self.topo_order = find_topo_sort(self.eval_node_list)
        self.node_to_shape_map = None
        self.feed_shapes = None
        self.top_control_queue = top_control_queue
        self.top_message_queue = top_message_queue
        self.control_queue = queue.Queue()
        self.have_done_queue = queue.Queue()
        self.memoryManagerController = MemoryManagerController(self.control_queue,
                                                               self.have_done_queue)
        self.memoryManagerController.start()

        self.cudaStream = gpu_op.create_cudaStream()
        self.cudnnHandle = gpu_op.create_cudnnHandle(self.cudaStream)
        self.cublasHandle = gpu_op.create_cublasHandle(self.cudaStream)

        # 按照拓扑排序设定index
        for i in range(len(self.topo_order)):
            self.topo_order[i].index = i

        print("最后输出index：")
        for node in eval_node_list:
            print(node.index)

        # todo 此处hard code，后续需要修改
        self.ctx_cpu = ndarray.cpu(0)
        self.ctx_gpu = ndarray.gpu(0)

    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.op.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        self.node_to_shape_map = dict(feed_shapes)

        for idx, node in enumerate(self.topo_order):
            if node in self.node_to_shape_map:
                continue
            input_shapes = [self.node_to_shape_map[i] for i in node.inputs]
            assert None not in input_shapes
            self.node_to_shape_map[node] = node.op.infer_shape(node, input_shapes, self.cudnnHandle)

    def memory_plan(self, feed_shapes):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.

        Implementation note:
        Option 1: Alloc a ndarray.NDArray per node that persists across run()
        Option 2: Implement a memory pool to reuse memory for nodes of same
                shapes. More details see Lecture 7.

        For both options, self.node_to_arr_map stores node->NDArray mapping to
        allow mapping to persist across multiple executor.run().

        Hint: use ndarray.empty(shape, ctx=self.ctx) to allocate NDArray.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        # self.node_to_arr_map = {}
        # for node in self.topo_order:
        #     self.node_to_arr_map[node] = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx_cpu)

        assert False

    def run(self, feed_dict, convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array

        Returns
        -------
        A list of values for nodes in eval_node_list. NDArray or np.ndarray.
        """

        def are_feed_shapes_equal(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            unmatched_item = set(sa.items()) ^ set(sb.items())
            return len(unmatched_item) == 0

        # Assume self.ctx is None implies numpy array and numpy ops.
        global index_to_gpu_map
        global index_to_cpu_map
        global index_to_cpu_flag
        index_to_gpu_map = {}
        index_to_cpu_flag = {}
        for node, value in feed_dict.items():
            # convert values to ndarray.NDArray if necessary
            # 源代码会在此处将所有CPU的内容引入GPU，为了自定义，禁用自动引入的功能，改为手动引入
            if isinstance(value, np.ndarray):
                index_to_gpu_map[node.index] = ndarray.array(value, ctx=self.ctx_cpu)
            elif isinstance(value, ndarray.NDArray):
                index_to_gpu_map[node.index] = value
            else:
                assert False, "feed_dict value type not supported"

        # collect shapes for all placeholders
        feed_shapes = {}
        for i in index_to_gpu_map:
            feed_shapes[self.topo_order[i]] = index_to_gpu_map[self.topo_order[i].index].shape

        if self.feed_shapes is None:
            # todo 向上层返回需要的信息
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            for node in self.node_to_shape_map:
                index_to_cpu_map[node.index] = ndarray.empty(self.node_to_shape_map[node], self.ctx_cpu)
            return_list = []
            for node in self.topo_order:
                node_inputs = []
                for node_input in node.inputs:
                    node_inputs.append(node_input.index)
                node_size = np.prod(self.node_to_shape_map[node]) * 4
                print("node" + str(node.index) + " size: " + str(node_size))
                # if len(self.node_to_shape_map[node]) == 1:
                #     node_size = self.node_to_shape_map[node][0] * 4
                # else:
                #     node_size = self.node_to_shape_map[node][0] * self.node_to_shape_map[node][1] * 4
                operation_name = node.op
                return_element = [node.index, node_inputs, node_size, operation_name]
                return_list.append(return_element)
            self.top_message_queue.put([0, return_list])
        else:
            return_list = []
            for i in range(len(self.topo_order)):
                return_element = (i, self.topo_order[i].runtime)
                return_list.append(return_element)
            self.top_message_queue.put([1, return_list])

        # infer shape if feed_shapes changed since last run
        # e.g. call run() on test data after trainng
        if not are_feed_shapes_equal(feed_shapes, self.feed_shapes):
            # todo not allowed to change when running
            assert False
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes

        # calculate started

        for node in self.topo_order:
            node.array_status = 0

        # todo 测试用
        have_got_global_message = False

        if not self.top_control_queue.empty():
            have_got_global_message = True
            print("get control message")
            # todo 解析从上游传入的控制信息。

            top_swap_list, top_release_list, top_recomputation_list = self.top_control_queue.get()

            # 顺序为(start_node, start_node_type, start_time, node_id, move_to_gpu)
            # 此处保证start_time按照顺序排布

            for control_node in self.topo_order:
                control_node.control_message_in = []
                control_node.control_message_in_time = 0
                control_node.control_message_out = []
                control_node.control_message_out_time = 0
                # wait_time, node_id, move_to_gpu
                control_node.recompute_list = []
                control_node.release_list = []

            for swap_message in top_swap_list:
                node_index = swap_message[0]
                start_time = swap_message[1]
                start_node_id = swap_message[2]
                move_to_gpu = swap_message[3]

                start_node = self.topo_order[start_node_id]
                if start_node.control_message_out_time == 0:
                    start_node.control_message_out_time = start_time
                    start_node.control_message_out.append((start_time, node_index, move_to_gpu))
                else:
                    start_node.control_message_out.append(
                        (start_time - start_node.control_message_out_time, node_index, move_to_gpu))
                    start_node.control_message_out_time = start_time

            for release_message in top_release_list:
                start_node_id = release_message[0]
                node_id = release_message[1]

                start_node = self.topo_order[start_node_id]
                start_node.release_list.append(node_id)

            # print(top_swap_list)
            # print(top_release_list)
            # print(top_recomputation_list)
            print("swap list")
            for node in self.topo_order:
                print(node.control_message_out)
            print("recompute list")
            for node in self.topo_order:
                print(node.recompute_list)
            print("release list")
            for node in self.topo_order:
                print(node.release_list)
            print("update control message")

        # Traverse graph in topo order and compute values for all nodes.
        for node in self.topo_order:
            if have_got_global_message:
                print(node.index)

            if node.index in index_to_gpu_map:
                # Skip placeholder nodes. Values already provided by feed_dict.
                # 找出feed_dict中已经包含的ndarray
                node.array_status = 1
                continue

            input_vals = []

            for recompute_index in node.recompute_list:
                # todo  加入重计算的过程,重计算在被动swap in之前
                recompute_node = self.topo_order[recompute_index]
                recompute_inputs = []
                for input_node in recompute_node.inputs:
                    recompute_inputs.append(node_to_gpu_map[input_node])
                recompute_ndarray = ndarray.empty(self.node_to_shape_map[recompute_node], self.ctx_gpu)
                recompute_node.array_status = 1
                recompute_node.op.compute(recompute_node, recompute_inputs, recompute_ndarray, False)
                index_to_gpu_map[recompute_node.index] = recompute_ndarray

            for n in node.inputs:
                if index_to_gpu_map[n.index] is None:
                    print("when computing " + str(node.index) + " passive import " + str(n.index))
                    # todo 考虑如何被动进行swap in
                    assert index_to_cpu_flag[n.index], "输入tensor不在cpu上"
                    node_ndarray_new = ndarray.empty(self.node_to_shape_map[n], self.ctx_gpu)
                    index_to_cpu_map[n.index].copyto(node_ndarray_new, self.cudaStream)
                    index_to_gpu_map[n.index] = node_ndarray_new
                    n.array_status = 1
                input_vals.append(index_to_gpu_map[n.index])

            # input_vals = [node_to_gpu_map[n] for n in node.inputs]
            node_val = ndarray.empty(self.node_to_shape_map[node], self.ctx_gpu)
            node.array_status = 1

            # node_val is modified in-place whether np.ndarray or NDArray
            # node_val是开辟出来用来保存每一个的节点的计算的结果的，计算成功后会放入node_to_val中

            for control_message in node.control_message_in:
                wait_time = control_message[0]
                node_id = control_message[1]
                move_to_gpu = control_message[2]
                if move_to_gpu:
                    self.control_queue.put((wait_time, node_id, move_to_gpu))
                else:
                    self.control_queue.put((wait_time, node_id, move_to_gpu))

            t1 = datetime.datetime.now()
            node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle, self.cudaStream, False)
            t2 = datetime.datetime.now()
            node.runtime = (t2 - t1).microseconds / 1000
            # print(node.index)

            # print(node.index)
            index_to_gpu_map[node.index] = node_val

            for control_message in node.control_message_out:
                wait_time = control_message[0]
                node_id = control_message[1]
                move_to_gpu = control_message[2]
                if move_to_gpu:
                    self.control_queue.put((wait_time, node_id, move_to_gpu))
                else:
                    self.control_queue.put((wait_time, node_id, move_to_gpu))

                # # todo 仅用于测试
                # self.have_done_queue.get(block=True)
                # print("swap end")

            for release_message in node.release_list:
                index_to_gpu_map[release_message] = None
                self.topo_order[release_message].array_status = 0

        # Collect node values.
        # print("success one batch")
        return [index_to_gpu_map[n.index] for n in self.eval_node_list]


def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad
        input_grads_list = node.op.gradient(node, output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    #  print(grad_node_list[0])
    return grad_node_list


##################
# Helper Methods #
##################


def find_topo_sort(node_list):
    """Given a list of nodes, return a topo ordering of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a
    topological sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        #  print(node.name)
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    #
    # if isinstance(node, list):
    #     print(node[0])
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum func to avoid creating redundant nodes in Python sum func."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


def broadcast_rule(shape_a, shape_b):
    """Return output shape of broadcast shape_a, shape_b.
    e.g. broadcast_rule((3,2), (4,3,2))
    returns output_shape = (4,3,2)

    Check out explanations and more examples at
    https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html
    http://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/
    """
    assert (isinstance(shape_a, tuple))
    assert (isinstance(shape_b, tuple))
    if len(shape_a) > len(shape_b):
        longer_shape, shorter_shape = shape_a, shape_b
    else:
        longer_shape, shorter_shape = shape_b, shape_a
    len_diff = len(longer_shape) - len(shorter_shape)
    for i in range(len_diff):
        # pad with leading 1s
        shorter_shape = (1,) + shorter_shape
    assert len(shorter_shape) == len(longer_shape)
    output_shape = list(longer_shape)
    for i in range(len(output_shape)):
        assert (shorter_shape[i] == longer_shape[i]) \
               or (shorter_shape[i] == 1) \
               or (longer_shape[i] == 1)
        output_shape[i] = max(shorter_shape[i], longer_shape[i])
    return tuple(output_shape)