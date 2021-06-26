from __future__ import absolute_import
import time
import numpy as np
from pycode.tinyflow  import ndarray, gpu_op, memoryManager
import random
import queue
from pycode.tinyflow import autodiff_vdnn as ad
import datetime, queue, os, time, sys

class TrainExecutor(object):
    """Executor computes values for given set of nodes in computation graph."""


    def __init__(self, targetloss, learning_rate=0.001, ctx=ndarray.gpu(0)):

        self.b1 = 0.9
        self.b2 = 0.999
        self.e = 0.00000001
        self.b1t = [0.9]
        self.b2t = [0.999]
        self.targetloss = targetloss
        self.learning_rate = learning_rate
        self.Variable_node_list = get_Variable_node_list(self.targetloss)

        self.Variable_node_list.reverse()
        self.Variable_node_grad_list = ad.gradients(self.targetloss, self.Variable_node_list)#反向node

        self.eval_node_list,self.mv,self.Variable_node_to_mv = getcomputelist(self.Variable_node_list,self.Variable_node_grad_list, self.b1, self.b2, self.b1t, self.b2t, self.e, self.learning_rate)#其内存还是Variable，但是换了个点
        self.ctx = ctx
        # 根据这个topo_order算
        self.topo_order = ad.find_topo_sort(self.eval_node_list)
        self.topo_order = swapadam(self.topo_order)


        #存node的shape
        self.node_to_shape_map = None
        #node和其对应的value，这里value自己定义,或者node本身可以判断状态
        self.node_to_arr_map = {}

        #初始化变量的np
        self.Variable_node_np_value = None

        #计算必要的资源
        self.cudaStream = gpu_op.create_cudaStream()
        self.cudnnHandle = gpu_op.create_cudnnHandle(self.cudaStream)
        self.cublasHandle = gpu_op.create_cublasHandle(self.cudaStream)


        #是否是第一次run
        self.isfirstrun = 0
        self.isc = 0

        # 按照拓扑排序设定index
        for i in range(len(self.topo_order)):
            self.topo_order[i].index = i

        self.will_do_queue = queue.Queue()
        self.have_done_queue = queue.Queue()
        self.memoryManager = memoryManager.MemoryManager(self.will_do_queue, self.have_done_queue)
        self.memoryManager.start()

        #日志记录
        self.start_finish_time = 0
        self.hit_count = 0
        self.swap_count = 0
        self.node_order = []


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
            print(node, self.node_to_shape_map[node])

    #放出变量的np字典
    def init_Variable(self, feed_dict):
        self.Variable_node_np_value = feed_dict
        for node in self.Variable_node_list:
            nodem = self.Variable_node_to_mv[node][0]
            nodev = self.Variable_node_to_mv[node][1]
            self.Variable_node_np_value[nodem] = np.zeros(feed_dict[node].shape)
            self.Variable_node_np_value[nodev] = np.zeros(feed_dict[node].shape)

    def find_prefetch_layer(self, currId):
        for i in range(currId + 1, len(self.topo_order)):
            node = self.topo_order[i]
            for n in node.inputs:
                if n.array_status == 0:
                    return i
            if node.is_conv == 1 or node.is_conv == 2 or node.issgd == 1:
                return -1
        return -1

    #feed_dict为np数组
    def run(self, feed_dict, Accuracy_node = None ,convert_to_numpy_ret_vals=False):


        if self.isfirstrun == 0:


            #第一次,把变量一起初始化了
            feed_dict.update(self.Variable_node_np_value)

            # 先确定shape

            #input的shape
            feed_shapes = {}
            for node, value in feed_dict.items():
                feed_shapes[node] = value.shape

            #把shape放进self.node_to_shape_map
            self.infer_shape(feed_shapes)

            #存已经被计算过的node
            node_computed = set()

            # 日志记录
            self.start_finish_time = datetime.datetime.now()
            self.node_order.append("topo_order:")
            for i in range(len(self.topo_order)):
                self.node_order.append("index:" + str(i) + "\t" + self.topo_order[i].name)
            self.node_order.append("\nrun:")

            #开始运行
            for i in range(len(self.topo_order)):
                # time.sleep(3)
                node = self.topo_order[i]
                self.node_order.append("index:" + str(i) + "\t" + node.name + "\ttime:" + str(datetime.datetime.now()))

                # print("")
                # print(node, " ", node.index)

                # count = 0
                # for node1 in self.topo_order:
                #     if node1 in self.node_to_arr_map:
                #         if self.node_to_arr_map[node1] != None:
                #             if ndarray.is_gpu_ctx(self.node_to_arr_map[node1].ctx):
                #                 count = count + 1
                # count2 = 0
                # for node2 in self.will_do_queue.:
                #     if ndarray.is_gpu_ctx(node2[1].ctx):
                #         count2 = count2 + 1
                # print(count)
                #
                # for ii in node.inputs:
                #     print(" input:" ,ii, ii.index)

                #已经被计算过了
                if node in node_computed:
                    continue
                #是inputs
                if node in feed_dict.keys():
                    #申请空间

                    ret = ndarray.array(feed_dict[node], ctx=self.ctx)
                    while isinstance(ret, int):
                        # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                        assert 0
                        #解决了再声明内存
                        ret = ndarray.array(feed_dict[node], ctx=self.ctx)
                    #此时ret为ndarray
                    #value都存在self.node_to_arr_map
                    self.node_to_arr_map[node] = ret
                    node_computed.add(node)
                    node.array_status = 1  # on GPU
                    continue


                #不是SgdOp,申请内存
                if node.issgd == 0:
                    #给这个点申请内存
                    # 申请空间
                    # pynvml.nvmlInit()
                    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示显卡标号
                    # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    # print(meminfo.total / 1024 ** 2)  # 总的显存大小
                    # print(meminfo.used / 1024 ** 2)  # 已用显存大小
                    # print(meminfo.free / 1024 ** 2)  # 剩余显存大小

                    ret = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)
                    while isinstance(ret, int):
                        # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                        # print(self.node_to_shape_map[node])
                        #
                        # pynvml.nvmlInit()
                        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示显卡标号
                        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        # print(meminfo.total / 1024 ** 2)  # 总的显存大小
                        # print(meminfo.used / 1024 ** 2)  # 已用显存大小
                        # print(meminfo.free / 1024 ** 2)  # 剩余显存大小
                        # ret = ndarray.empty((1, 1, 1, 1), ctx=self.ctx)
                        # print(ret)
                        assert 0
                        # 解决了再声明内存
                        ret = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)
                    # 此时ret为ndarray
                    # value都存在self.node_to_arr_map
                    self.node_to_arr_map[node] = ret
                    node.array_status = 1  # on GPU
                else:
                    # 是SgdOp,不申请内存
                    self.node_to_arr_map[node] = None

                prefeth_node_index = self.find_prefetch_layer(i)  # 预取最近的层的输入
                if prefeth_node_index != -1:
                    prefeth_node = self.topo_order[prefeth_node_index]
                    for n in prefeth_node.inputs:
                        if n.array_status == 0 or n.array_status == 3:  # on CPU
                            # print("pre node index", prefeth_node_index)
                            # print("input.index", n.index)
                            self.will_do_queue.put((n.index, self.node_to_arr_map[n]))
                            n.array_status = 2  # CPU to GPU
                            self.hit_count = self.hit_count + 1
                            self.node_order.append("swap_in\t" + "index:" + str(n.index)
                                                   + "\t" + n.name + "\ttime:" + str(datetime.datetime.now()))

                for n in node.inputs:   # 计算前确保输入已经放在GPU
                    if n.array_status == 0:  # on CPU

                        print("显存", self.topo_order[n.index], n.index)
                        self.will_do_queue.put((n.index, self.node_to_arr_map[n]))
                        n.array_status = 2  # CPU to GPU
                        self.node_order.append("swap_in\t" + "index:" + str(n.index)
                                               + "\t" + n.name + "\ttime:" + str(datetime.datetime.now()))


                for n in node.inputs:
                    while n.array_status != 1:
                        (node_index, node_val) = self.have_done_queue.get(block=True)  # get the node from GPU
                        print("成功搬到显存", self.topo_order[node_index], node_index)
                        self.node_to_arr_map[self.topo_order[node_index]] = node_val
                        if ndarray.is_gpu_ctx(node_val.ctx):
                            self.topo_order[node_index].array_status = 1
                            self.swap_count = self.swap_count + 1
                            self.node_order.append("finish_swap_in\t"
                                                   + "index:" + str(node_index) + "\t" + self.topo_order[node_index].name
                                                   + "\ttime:" + str(datetime.datetime.now()))
                        else:
                            self.topo_order[node_index].array_status = 0
                            self.node_order.append("finish_swap_out\t"
                                                   + "index:" + str(node_index) + "\t" + self.topo_order[node_index].name
                                                   + "\ttime:" + str(datetime.datetime.now()))
                    # if node.name == "AdamOp":
                    #     print("input_node", n)
                    #     print(np.prod(self.node_to_shape_map[node]) * 4)



                #放inputs的ndarray，
                input_vals = []
                for input_node in node.inputs:
                    #此时要保证在gpu中
                    if node.inputs:
                        input_vals.append(self.node_to_arr_map[input_node])

                if node.is_conv == 1 or node.is_conv == 2:  # 如果该层是卷积正向,在计算前准备将当前层的输入移到CPU,(因为是node是dfs顺序,所以可以直接卸载当前层输入)
                    for n in node.inputs:
                        if n.isw == 1:  # 只卸载Xs,不卸载参数
                            continue
                        if n.array_status == 1:  # on GPU
                            print("内存", self.topo_order[n.index], n.index)
                            self.will_do_queue.put((n.index, self.node_to_arr_map[n]))
                            n.array_status = 3  # GPU to CPU
                            self.node_to_arr_map[n] = None
                        self.node_order.append("swap_out\t" + "index:" + str(n.index)
                                               + "\t" + n.name + "\ttime:" + str(datetime.datetime.now()))


                #除了SgdOp，其他的点此时要保证在gpu中
                node_val = self.node_to_arr_map[node]


                memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle, self.cudaStream)
                while memorytoSaving != 0:
                    #不等于0意味着运行需要的临时内存不够，memorytoSaving是申请失败的cudamalloc（，size）的size
                    #这里运行被动模式
                    assert 0
                    #解决了重新计算
                    memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle, self.cudaStream)

                #此点被计算过了
                node_computed.add(node)

                if node.is_conv == 1 or node.is_conv == 2:  # 计算后确保卷积的输入已经移到CPU
                    for n in node.inputs:
                        if n.isw == 1:  # 只卸载Xs,不卸载参数
                            continue
                        while n.array_status != 0:
                            (node_index, node_val) = self.have_done_queue.get(block=True)
                            self.node_to_arr_map[self.topo_order[node_index]] = node_val
                            if ndarray.is_gpu_ctx(node_val.ctx):
                                self.topo_order[node_index].array_status = 1
                                self.swap_count = self.swap_count + 1
                                self.node_order.append("finish_swap_in\t"
                                                       + "index:" + str(node_index) + "\t" + self.topo_order[node_index].name
                                                       + "\ttime:" + str(datetime.datetime.now()))
                            else:
                                self.topo_order[node_index].array_status = 0
                            print("成功搬到内存", self.topo_order[node_index], node_index, self.topo_order[node_index].array_status)

            # 不是第一次了
            self.isfirstrun = 1

            #把结果输出了： [loss,变量按网络顺序],这里只是输出value，并不保证一定在gpu中
            #但是如果这里value是None的话，他会报错
            result_output = [self.node_to_arr_map[self.targetloss]]
            re_var = []
            for node in self.Variable_node_list:
                re_var.append(self.node_to_arr_map[node])
            re_var.reverse()
            result_output = result_output + re_var
            #结果，计算正确率
            if Accuracy_node !=None:
                result_output.append(self.node_to_arr_map[Accuracy_node])

            #adam更新参数
            self.b1t[0] = self.b1t[0] * self.b1
            self.b2t[0] = self.b2t[0] * self.b2

            return result_output



        else:

            # 存已经被计算过的node
            node_computed = set()
            # 开始运行

            # for node in self.topo_order:
            for i in range(len(self.topo_order)):
                node = self.topo_order[i]
                self.node_order.append("index:" + str(i) + "\t" + node.name + "\ttime:" + str(datetime.datetime.now()))


                # 已经被计算过了
                if node in node_computed:
                    continue
                # 是inputs
                if node in feed_dict.keys():

                    if ndarray.is_gpu_ctx(self.node_to_arr_map[node].ctx):
                        self.node_to_arr_map[node]._sync_copyfrom(feed_dict[node])
                    else:
                        # 没在GPU中,重新在GPU申请空间
                        ret = ndarray.array(feed_dict[node], ctx=self.ctx)
                        while isinstance(ret, int):
                            # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                            assert 0
                            # 解决了再声明内存
                            ret = ndarray.array(feed_dict[node], ctx=self.ctx)
                        # 此时ret为ndarray
                        # value都存在self.node_to_arr_map
                        self.node_to_arr_map[node] = ret

                        node.array_status = 1  # on GPU


                    node_computed.add(node)
                    continue

                # 如果node是变量，不用管
                if node in self.Variable_node_list:
                    continue
                # 如果node是adam要用的变量，不用管
                if node in self.mv:
                    continue

                #不是sgdop的中间点
                if node.issgd == 0:

                    # 在gpu中，可以直接拿来用，直接pass
                    if ndarray.is_gpu_ctx(self.node_to_arr_map[node].ctx):
                        pass
                    else:
                        # 不在gpu中，生成新的empty
                        # 申请空间
                        ret = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)
                        while isinstance(ret, int):
                            # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                            assert 0
                            # 解决了再声明内存
                            ret = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)
                        # 此时ret为ndarray
                        # value都存在self.node_to_arr_map
                        self.node_to_arr_map[node] = ret

                        node.array_status = 1

                prefeth_node_index = self.find_prefetch_layer(i)  # 预取最近的层的输入
                if prefeth_node_index != -1:
                    prefeth_node = self.topo_order[prefeth_node_index]
                    for n in prefeth_node.inputs:
                        # print("pre node index", prefeth_node_index)
                        # print("pre.index", n.index)
                        if n.array_status == 0:  # on CPU
                            self.will_do_queue.put((n.index, self.node_to_arr_map[n]))
                            n.array_status = 2  # CPU to GPU
                            self.hit_count = self.hit_count + 1
                            self.node_order.append("swap_in\t" + "index:" + str(n.index)
                                                   + "\t" + n.name + "\ttime:" + str(datetime.datetime.now()))

                for n in node.inputs:   #计算前确保输入已经放在GPU
                    if n.array_status == 0:  # on CPU
                        # print("n.index", n.index)
                        self.will_do_queue.put((n.index, self.node_to_arr_map[n]))
                        n.array_status = 2  # CPU to GPU
                        self.node_order.append("swap_in\t" + "index:" + str(n.index)
                                               + "\t" + n.name + "\ttime:" + str(datetime.datetime.now()))
                for n in node.inputs:
                    while n.array_status != 1:
                        (node_index, node_val) = self.have_done_queue.get(block=True)  # get the node from GPU
                        self.node_to_arr_map[self.topo_order[node_index]] = node_val
                        if ndarray.is_gpu_ctx(node_val.ctx):
                            self.topo_order[node_index].array_status = 1
                            self.swap_count = self.swap_count + 1
                            self.node_order.append("finish_swap_in\t"
                                                   + "index:" + str(node_index) + "\t" + self.topo_order[node_index].name
                                                   + "\ttime:" + str(datetime.datetime.now()))
                        else:
                            self.topo_order[node_index].array_status = 0
                            self.node_order.append("finish_swap_out\t"
                                                   + "index:" + str(node_index) + "\t" + self.topo_order[node_index].name
                                                   + "\ttime:" + str(datetime.datetime.now()))


                # 放inputs的ndarray，
                input_vals = []
                for input_node in node.inputs:
                    # 此时要保证在gpu中
                    input_vals.append(self.node_to_arr_map[input_node])

                if node.is_conv == 1 or node.is_conv == 2:  # 如果该层是卷积正向或反向，在计算前准备将当前层的输入移到CPU
                    for n in node.inputs:
                        if node.isw == 1:
                            continue
                        if n.array_status == 1:  # on GPU
                            self.will_do_queue.put((n.index, self.node_to_arr_map[n]))
                            n.array_status = 3  # GPU to CPU
                            self.node_to_arr_map[n] = None
                        self.node_order.append("swap_out\t" + "index:" + str(n.index)
                                               + "\t" + n.name + "\ttime:" + str(datetime.datetime.now()))


                # 除了SgdOp，其他的点此时要保证在gpu中
                node_val = self.node_to_arr_map[node]

                memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle, self.cudaStream)
                while memorytoSaving != 0:
                    # 不等于0意味着运行需要的临时内存不够，memorytoSaving是申请失败的cudamalloc（，size）的size
                    # 这里运行被动模式
                    assert 0
                    # 解决了重新计算
                    memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle, self.cudaStream)

                # 此点被计算过了
                node_computed.add(node)

                if node.is_conv == 1 or node.is_conv == 2:  # 计算后确保卷积的输入已经移到CPU
                    for n in node.inputs:
                        if node.isw == 1:
                            continue
                        while n.array_status != 0:
                            # print("node.index", node.index)
                            # print("n.index", n.index)
                            # print("n.name", n.name)
                            (node_index, node_val) = self.have_done_queue.get(block=True)
                            # print("node_index", node_index)
                            # print("node_val", node_val)
                            self.node_to_arr_map[self.topo_order[node_index]] = node_val
                            if ndarray.is_gpu_ctx(node_val.ctx):
                                self.topo_order[node_index].array_status = 1
                                self.swap_count = self.swap_count + 1
                                self.node_order.append("finish_swap_in\t"
                                                       + "index:" + str(node_index) + "\t" + self.topo_order[node_index].name
                                                       + "\ttime:" + str(datetime.datetime.now()))
                            else:
                                self.topo_order[node_index].array_status = 0
                                self.node_order.append("finish_swap_out\t"
                                                       + "index:" + str(node_index) + "\t" + self.topo_order[node_index].name
                                                       + "\ttime:" + str(datetime.datetime.now()))

            # 把结果输出了： [loss,变量按网络顺序],这里只是输出value，并不保证一定在gpu中
            # 但是如果这里value是None的话，他会报错
            result_output = [self.node_to_arr_map[self.targetloss]]
            re_var = []
            for node in self.Variable_node_list:
                re_var.append(self.node_to_arr_map[node])
            re_var.reverse()
            result_output = result_output + re_var
            # 结果，计算正确率
            if Accuracy_node != None:
                result_output.append(self.node_to_arr_map[Accuracy_node])

            # adam更新参数
            self.b1t[0] = self.b1t[0] * self.b1
            self.b2t[0] = self.b2t[0] * self.b2
            return result_output

    def get_start_finish_time(self):
        return self.start_finish_time

    def get_hit(self):
        return self.hit_count, self.swap_count

    def get_node_order(self):
        return self.node_order








def get_Variable_node_list(node):

    visited = set()
    Variable_order = []
    Variable_sort_dfs(node, visited, Variable_order)
    return Variable_order



def Variable_sort_dfs(node, visited, Variable_order):
    """Post-order DFS"""
    #
    # if isinstance(node, list):
    #     print(node[0])
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        Variable_sort_dfs(n, visited, Variable_order)

    if node.isw == 1:
        Variable_order.append(node)



def getcomputelist(Variable_node_list, Variable_node_grad_list, b1, b2, b1t, b2t, e,learning_rate):

    computelist = []
    mv = []
    Variable_node_to_mv = {}
    for i in range(len(Variable_node_list)):
        m = ad.Variable(Variable_node_list[i].name+'m')
        v = ad.Variable(Variable_node_list[i].name+'v')
        mv.append(m)
        mv.append(v)
        Variable_node_to_mv[Variable_node_list[i]] = (m,v)
        adamnode = ad.adam_op(Variable_node_list[i],m,v,Variable_node_grad_list[i], b1, b2, b1t, b2t, e, learning_rate)
        adamnode.issgd = 1#代表不用为这个点加内存
        computelist.append(adamnode)

    return computelist,mv,Variable_node_to_mv




def swapadam(topoorder):
    for i in range(len(topoorder)):
        if topoorder[i].issgd == 1 and topoorder[i].isw == 0:
            topoorder[i].isw = 3
            filter = topoorder[i].inputs[0]
            j = len(topoorder) - 1
            while j > i:
                if topoorder[j].issgd == 1:
                    j = j - 1
                    continue
                if filter in topoorder[j].inputs:

                    break
                j = j - 1

            tmp = topoorder[i]
            topoorder.remove(tmp)
            topoorder.insert(j,tmp)
    for i in range(len(topoorder)):
        print(i,topoorder[i])
    return topoorder







