import copy
from enum import Enum
import multiprocessing
import numpy as np
from functools import cmp_to_key
import plotly as py
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
from collections import defaultdict
import os
from pynvml import *
import time
import matplotlib
# matplotlib.use('Agg')
import numpy as np
import os
from keras import Model, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pynvml import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from matplotlib import cm
from tensorboard.plugins.hparams import keras
from tools import *

GPU = load_gpu()
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU}"
pyplt = py.offline.plot
PCIE_bandwidth = 12  # MB/ms
load_list = ['convolution_2d_forward_VALID', 'convolution_backward_filter_2d_VALID', 'convolution_backward_data_2d_VALID',
             'convolution_2d_forward_SAME', 'convolution_backward_filter_2d_SAME', 'convolution_backward_data_2d_SAME',
             'dropout_forward', 'dropout_backward', 'broadcast_to_NHWC',
             'broadcast_to_NCHW', 'reduce_sum_new_NHWC', 'reduce_sum_new_NCHW',
             'bn_forward_pre_activation', 'bn_backward_pre_activation', 'activation_forward_relu',
             'activation_backward_relu', 'activation_forward_softmax', 'activation_backward_softmax',
             'pooling_2d_forward_max', 'pooling_2d_backward_max', 'pooling_2d_forward_mean',
             'pooling_2d_backward_mean', 'matrix_multiply', 'matrix_elementwise_multiply_by_const', 'matrix_elementwise_add',
             'array_set', 'concat_forward', 'concat_a_backward',
             'concat_b_backward', 'sgd_update', 'cross', 'cross_backward', 'adam_mv', 'adam_compute']


class TaskType(Enum):
    swap_out = 0
    swap_in = 1


class AccessType(Enum):
    output = 0
    input = 1


class Tensor:
    def __init__(self, tensor_id, job_id, size, shape, recomputation_time, source_tensors=None, is_parameter=False, is_input_or_output=False):
        self.tensor_id = tensor_id
        self.job_id = job_id
        self.size = size
        self.swap_time = self.size / PCIE_bandwidth
        self.source_tensors = source_tensors if source_tensors is not None else []
        self.recomputation_time = recomputation_time
        self.recomputation_metric = self.size / self.recomputation_time
        self.is_parameter = is_parameter
        self.shape = shape
        if self.is_parameter or is_input_or_output:
            self.in_gpu_at_beginning = True
        else:
            self.in_gpu_at_beginning = False

    def __repr__(self):
        return f'tensor_id:{self.tensor_id}, job_id":{self.job_id}, size:{self.size}'


class TensorAccess:
    def __init__(self, tensor, time, run_time, access_type, operation_id, operation_name):
        self.tensor = tensor
        self.access_id = None
        self.start_time = None
        self.end_time = None
        self.time = time
        self.run_time = run_time
        self.access_type = access_type
        if self.access_type == AccessType.output:
            self.end_time = self.time
            self.start_time = self.time - self.run_time
        else:
            self.start_time = self.time
            self.end_time = self.time + self.run_time
        self.release_flag = False
        self.operation_id = operation_id
        self.operation_name = operation_name

    def to_tuple(self):
        return (self.tensor.tensor_id, self.time)

    def __repr__(self):
        return f'id={self.tensor.tensor_id}, time={self.time}, access_type={self.access_type}'


class SwapTask(object):
    '''Date weighted interval'''

    def __init__(self, tensor, time, time_cost, task_type: TaskType, front_boundary=None, back_boundary=None):
        self.tensor = tensor
        self.time_cost = time_cost
        self.data_type = np.float64
        self.task_type = task_type
        self.swap_task_id = None
        assert not (front_boundary is None and back_boundary is None)
        # 最早开始时间
        self.front_boundary = front_boundary
        # 最晚结束时间
        self.back_boundary = back_boundary
        self.time = time
        self.execute_time = None
        self.execute_ref = None

    @property
    def start_time(self):
        return self.start_time_

    @start_time.setter
    def start_time(self, value):
        self.start_time_ = value
        if self.task_type == TaskType.swap_out:
            self.time = self.start_time_

    @property
    def end_time(self):
        return self.end_time_

    @end_time.setter
    def end_time(self, value):
        self.end_time_ = value
        if self.task_type == TaskType.swap_in:
            self.time = self.end_time_

    @classmethod
    def from_access(cls, access: TensorAccess, weight, task_type, front_boundary=None, back_boundary=None):
        return cls(access.tensor, weight, access.time, access.tensor.swap_time, task_type, front_boundary=front_boundary, back_boundary=back_boundary)

    def __repr__(self):
        return f'id={self.tensor}, type={self.task_type}, start_time={self.start_time}, end_time={self.end_time}'


def numpy_ewma_vectorized(data, window):
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = data.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


debug_num = 0


def create_model(n):
    model = Sequential()
    model.add(Dense(units=2048, activation='tanh', input_dim=n))
    model.add(Dense(units=2048, activation='tanh'))
    model.add(Dense(units=1, activation='relu'))
    return model


def load(opname, n):
    model = create_model(n)
    model.load_weights('model_parameter/' + opname + '_model.hdf5', by_name=True, skip_mismatch=True)
    return model


def load_all_model():
    global models
    global load_list
    old_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for op_name in load_list:
        scaler = []
        with open('../../data_bn/' + op_name + '_mean_and_std.txt', mode='r') as f:
            content = f.readlines()
            for c in content:
                c = c.strip()
                c = c.split(' ')
                scaler.append([float(c[0]), float(c[1])])
            model = load(op_name, len(scaler) + 1)
        models[op_name] = (scaler, model)
    os.environ["CUDA_VISIBLE_DEVICES"] = old_gpu


def get_predicted_execution_time(op_name, inputs_of_model, logged_time: list):
    # if len(logged_time) > 1:
    #     return logged_time[1]
    # else:
    #     return 50
    global models
    old_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    # 使用CPU进行推理
    gpu_usage = nvmlDeviceGetUtilizationRates(handle).gpu
    inputs_of_model.append(gpu_usage)
    scaler, model = models[op_name]
    assert len(inputs_of_model) == len(scaler)
    for i, mean, std in enumerate(scaler):
        inputs_of_model[i] = (inputs_of_model[i] - mean) / std
    predicted_time = model.predict(inputs_of_model)
    if len(logged_time) > 0:
        predicted_time = [predicted_time]
        predicted_time.extend(logged_time)
        predicted_time = numpy_ewma_vectorized(np.array(predicted_time), 3)
    os.environ["CUDA_VISIBLE_DEVICES"] = old_gpu
    return predicted_time


def liveness_analysis(tensor_access_list):
    global tensor_access_by_tensor
    # 活跃性分析结果生成
    for job_id in range(len(tensor_access_list)):
        tmp = set()
        for i in range(len(tensor_access_list[job_id]) - 1, -1, -1):
            tensor_access = tensor_access_list[job_id][i]
            if tensor_access.tensor not in tmp and len(tensor_access_by_tensor[tensor_access.tensor.job_id][tensor_access.tensor]) > 1:
                # 新生成的参数不会释放
                if tensor_access.operation_name != 'feed_dict' and tensor_access.tensor.is_parameter:
                    continue
                tmp.add(tensor_access.tensor)
                tensor_access.release_flag = True


def is_overlap(task: SwapTask, target: SwapTask):
    return task != target and (
            target.start_time < task.end_time < target.end_time or target.start_time < task.start_time < target.end_time or task.start_time < target.end_time < task.end_time or task.start_time < target.start_time < task.end_time)


def get_free_intervals(target_task, swap_schedule, key=0, asc=True):
    # 列出在可行区间内的所有空白时间区间，并按区间排序
    if target_task.back_boundary - target_task.front_boundary < target_task.time_cost:
        return []
    intervals = []
    for task in swap_schedule:
        if target_task.front_boundary < task.start_time < task.end_time < target_task.back_boundary:
            intervals.append((task.start_time, task.end_time))
        elif task.start_time < target_task.front_boundary < task.end_time < target_task.back_boundary:
            intervals.append((target_task.front_boundary, task.end_time))
        elif target_task.front_boundary < task.start_time < target_task.back_boundary < task.end_time:
            intervals.append((task.start_time, target_task.back_boundary))
        elif task.start_time < target_task.front_boundary < target_task.back_boundary < task.end_time:
            return []
    intervals = sorted(intervals, key=lambda x: x[0])
    not_occupied_intervals = []
    s = target_task.front_boundary

    for interval in intervals:
        if s < interval[0]:
            not_occupied_intervals.append((s, interval[0]))
        s = interval[1]
    if s < target_task.back_boundary:
        not_occupied_intervals.append((s, target_task.back_boundary))
    # 按照区间起点/终点排序
    not_occupied_intervals = sorted(not_occupied_intervals, key=lambda x: x[key], reverse=not asc)
    return not_occupied_intervals


def generate_swap_recomputation_release_order(tensor_access_by_tensor, swap_scheduler, recomputations, job_num):
    swap_orders = defaultdict(list)
    release_orders = defaultdict(list)
    recomp_orders = defaultdict(list)
    for job_id in range(job_num):
        # 按id排序
        tensor_accesses = sorted([i for tmp in tensor_access_by_tensor[job_id].values() for i in tmp], key=lambda x: x.tensor.tensor_id)
        # 按起始时间排序
        swap_tasks = sorted(swap_scheduler[job_id], key=lambda x: x.start_time)
        for i in range(len(swap_tasks)):
            swap_tasks[i].swap_task_id = i
        releases = []
        swaps = []
        recomps = []
        for access in tensor_accesses:
            if access.release_flag:
                releases.append((access.operation_id, access.tensor.tensor_id))
        release_orders[job_id] = releases
        for access in recomputations:
            recomps.append((access.operation_id, access.tensor.tensor_id, access.time))
        recomp_orders[job_id] = recomps
        for task in swap_tasks:
            # (task_id, node_id(tensor_id), start_time, start_node, move_to_gpu, start_node_type)
            swaps.append([task.tensor.tensor_id, task.execute_time, task.execute_ref.operation_id, 0 if task.task_type == TaskType.swap_out else 1, 1, task.start_time])
        swap_orders[job_id] = list(map(lambda x: x[:-1], sorted(swaps, key=lambda x: x[-1])))
    return release_orders, swap_orders, recomp_orders


def draw_all_task(tensor_access_by_tensor, swap_scheduler, job_num):
    for job_id in range(job_num):
        tmp = list(tensor_access_by_tensor[job_id].values())
        res = []
        for sub_list in tmp:
            res.extend(sub_list)
        draw(sorted(res, key=lambda x: x.start_time), swap_scheduler[job_id])


def get_max_memory_used(tensor_access_list, swap_tasks, swapped_out_tensor, recomputation_tensor, tensor_access_by_tensor, tensors):
    # 计算显存开销
    tmp = [tensor_access for tensor_access in tensor_access_list]
    tmp.extend(swap_tasks)

    def custom_cmp(x, y):
        if x.time < y.time:
            return -1
        elif x.time > y.time:
            return 1
        else:
            if x.start_time < y.start_time:
                return -1
            elif x.start_time > y.start_time:
                return 1
            else:
                return 0

    time_axis = sorted(tmp, key=cmp_to_key(custom_cmp))
    # occupied by handle, cudnn, cuda stream and cudart
    memory_used = 0
    max_memory_actual = float('-inf')
    in_gpu_tensors = set()
    max_memory_tensors = set()
    last_input_tensor_access = None
    max_last_access = None
    wait_to_be_released = []
    max_time = None
    foot_print = {}
    # 首先把输入的x，y以及所有没被swap out的参数载入显存，因为他们从上轮迭代结束时就一直在现存里面
    for tensor in tensors:
        if tensor.in_gpu_at_beginning and tensor not in swapped_out_tensor:
            in_gpu_tensors.add(tensor)
            memory_used += tensor.size
    for time_index, event in enumerate(time_axis):
        time = event.time
        for i in range(len(wait_to_be_released) - 1, -1, -1):
            access = wait_to_be_released[i]
            # 如果此刻时间已经过了释放时间，则释放该访问的附带影响
            if time >= access.end_time:
                wait_to_be_released.pop(i)
                assert access.tensor in recomputation_tensor or access.tensor in in_gpu_tensors
                if access.tensor in in_gpu_tensors:
                    memory_used -= access.tensor.size
                    in_gpu_tensors.remove(access.tensor)
        if isinstance(event, TensorAccess):
            if event.access_type == AccessType.output:
                # feed dict进入的参数已经swap in了，他们的feed dict不按照增加显存处理
                if not (event.tensor.is_parameter and event.operation_name == 'feed_dict') and event.tensor not in in_gpu_tensors:
                    memory_used += event.tensor.size
                    in_gpu_tensors.add(event.tensor)
            else:
                # 用完即释放的
                # input本身并不增加gpu使用，swap in增加
                if event.release_flag:
                    wait_to_be_released.append(event)
                else:
                    last_input_tensor_access = event
        elif isinstance(event, SwapTask):
            last_event = None
            for j in range(time_index - 1, -1, -1):
                if isinstance(time_axis[j], TensorAccess) and time_axis[j].end_time <= event.start_time:
                    last_event = time_axis[j]
                    break
            assert last_event is not None
            event.execute_ref = last_event
            event.execute_time = event.start_time - last_event.end_time
            if event.task_type == TaskType.swap_in:
                memory_used += event.tensor.size
                in_gpu_tensors.add(event.tensor)
            else:
                memory_used -= event.tensor.size
                in_gpu_tensors.remove(event.tensor)
        foot_print[time] = memory_used
        if memory_used > max_memory_actual:
            # max_memory_actual与是否有考虑价值无关，单纯计量峰值
            max_memory_actual = memory_used
            max_memory_tensors = copy.copy(in_gpu_tensors)
            max_last_access = copy.copy(last_input_tensor_access)
            max_time = time
    return max_memory_actual, max_memory_tensors, max_last_access, max_time, foot_print, time_axis


def run_global_memory_analysis(global_tensor_access, swap_tasks, swapped_out_tensor, recomputation_tensor, tensor_access_by_tensor):
    global global_tensors
    max_memory = 0
    max_memory_tensors = []
    last_input_accesses = []
    max_time = []
    foot_prints = []
    time_axis = []
    for job_id, tensor_accesses in enumerate(global_tensor_access):
        job_max_memory, job_max_memory_tensors, last_input_access, now_time, foot_print, t_axis = get_max_memory_used(tensor_accesses, swap_tasks[job_id], swapped_out_tensor, recomputation_tensor,
                                                                                                                      tensor_access_by_tensor[job_id], global_tensors[job_id])
        time_axis.append(t_axis)
        foot_prints.append(foot_print)
        max_memory_tensors.extend(job_max_memory_tensors)
        last_input_accesses.append(last_input_access)
        max_time.append(now_time)
        max_memory += job_max_memory
    return max_memory, max_memory_tensors, last_input_accesses, max_time, foot_prints, time_axis


def draw(tensor_access_list, swap_schedule):
    df = []
    id_color = {'OTA': 'rgb(255, 0, 102)', 'ITA': 'rgb(68, 114, 196)', 'Swap In': 'rgb(237, 137, 69)', 'Swap Out': 'rgb(112, 173, 71)'}
    for tensor_access in tensor_access_list:
        # input 蓝色，output红色
        df.append(dict(Task=f'tensor_id:{tensor_access.tensor.tensor_id}, size:{tensor_access.tensor.size}', Start=tensor_access.start_time, Finish=tensor_access.end_time,
                       Resource='OTA' if tensor_access.access_type == AccessType.output else 'ITA'))
    for task in swap_schedule:
        df.append(dict(Task=f'tensor_id:{task.tensor.tensor_id}, size:{task.tensor.size}', Start=task.start_time, Finish=task.end_time, Resource='Swap In' if task.task_type == TaskType.swap_in else 'Swap Out'))

    fig = ff.create_gantt(df, colors=id_color, index_col='Resource', group_tasks=True, show_colorbar=True, showgrid_x=True, showgrid_y=True, title=f'ratio={ratio}')
    fig['layout']['xaxis'].update({'type': None})
    fig.update_layout(
        height=900,
        width=1600,
    )
    pyplt(fig, filename=f'../../pic/job{tensor_access_list[0].tensor.job_id}.html', auto_open=True)


def try_swap_in(swap_in_task: SwapTask, swap_scheduler):
    # swap_in越晚越好，按结束时间降序排序
    free_intervals = get_free_intervals(swap_in_task, swap_scheduler[swap_in_task.tensor.job_id], 1, asc=False)
    succeed = False
    for interval in free_intervals:
        if interval[1] - interval[0] >= swap_in_task.time_cost:
            swap_in_task.end_time = interval[1]
            swap_in_task.start_time = swap_in_task.end_time - swap_in_task.time_cost
            swap_scheduler[swap_in_task.tensor.job_id].append(swap_in_task)
            succeed = True
            break
    if not succeed:
        return False
    else:
        return True


def can_next_input_access_swap_in(i, all_access_of_tensor, swap_out_task, swap_scheduler):
    # 至少将第一个访问swap in才算成功，后续的能换入的话，则把前一个的release_flag设为True
    access = all_access_of_tensor[i]
    swap_in_task = SwapTask(access.tensor, access.time, access.tensor.swap_time, TaskType.swap_in,
                            front_boundary=swap_out_task.end_time if swap_out_task.end_time > all_access_of_tensor[i - 1].end_time else all_access_of_tensor[i - 1].end_time,
                            back_boundary=access.time)
    return try_swap_in(swap_in_task, swap_scheduler)


def get_framework_info(info, logged_time, job_id):
    global global_tensors
    tensors = {}
    tensor_access_list = []
    global_time = 0
    parameter = []
    #   operation_id
    for output_tensor_id, input_tensor_id, output_tensor_size, operation_name, is_parameter, is_input_or_output, shape, inputs_of_model in info:
        # is_parameter: 生成的张量是否为参数
        # 输入的为Byte
        # 转换为MB
        output_tensor_size = output_tensor_size / 1000000
        input_tensors = []
        for tensor_id in input_tensor_id:
            input_tensor = tensors[tensor_id]
            input_tensors.append(input_tensor)
        time_cost = get_predicted_execution_time(operation_name, inputs_of_model, logged_time[output_tensor_id])
        output_tensor = Tensor(tensor_id=output_tensor_id, job_id=job_id, size=output_tensor_size, source_tensors=input_tensors, recomputation_time=time_cost, is_parameter=is_parameter, shape=shape)
        output_access = TensorAccess(tensor=output_tensor, time=global_time + time_cost, run_time=time_cost, access_type=AccessType.output, operation_id=output_tensor_id, operation_name=operation_name)
        tensor_access_list.append(output_access)
        tensors[output_tensor.tensor_id] = output_tensor
        if is_parameter:
            parameter.append(output_tensor)
        for tensor_id in input_tensor_id:
            input_tensor = tensors[tensor_id]
            input_access = TensorAccess(tensor=input_tensor, time=global_time, run_time=time_cost, access_type=AccessType.input, operation_id=output_tensor_id, operation_name=operation_name)
            tensor_access_list.append(input_access)
        global_time += time_cost

    tensors = list(tensors.values())
    global_tensors[job_id] = tensors
    tensor_access_list = sorted(tensor_access_list, key=lambda x: x.time)
    dic = defaultdict(list)
    for access in tensor_access_list:
        dic[access.tensor].append(access)
    for k, v in dic.items():
        dic[k] = sorted(v, key=lambda x: x.time)
    tensor_access_by_tensor[job_id] = dic

    swap_scheduler = []
    # 对参数进行swap in调度
    # earliest_swap = None
    earliest_time = float('inf')
    # 从最早的参数开始安排
    parameter = sorted(parameter, key=lambda x: dic[x][0].start_time)
    # for para in parameter:
    #     if not para.in_gpu:
    #         # 找到第一次访问
    #         first_access = dic[para][0]
    #         swap_in_task = SwapTask(para, first_access.time, first_access.tensor.swap_time, TaskType.swap_in, front_boundary=float('-inf'), back_boundary=first_access.start_time)
    #         # 对所有参数的第一次访问进行调度
    #         res = try_swap_in(swap_in_task, swap_scheduler)
    #         assert not res, f'swap in parameter:{para} failed'
    #         if swap_in_task.start_time < earliest_time:
    #             earliest_time = swap_in_task.start_time
    #             # earliest_swap = swap_in_task
    # # 重置时间轴
    # if earliest_time < 0:
    #     delta_time = -earliest_time
    #     tmp = copy.copy(swap_scheduler)
    #     tmp.extend(tensor_access_list)
    #     for event in tmp:
    #         event.time += delta_time
    #         event.end_time = event.end_time + delta_time
    #         event.end_time = event.end_time + delta_time

    return tensor_access_list, swap_scheduler, parameters


# 随机生成数据用的参数
times = 150
tensors = 50
time_scale = times
ratio = 1

# 全局变量
job_num = 0
global_tensor_access = [[]]
tensor_access_by_tensor = []
weight = 1
jobs_weights = []
# jobs_weight = [1, 1, 1, 1, 1]
total_memory = 0
enable_recomputation = True
global_graphs = []
global_tensors = {}
swap_scheduler = []
parameters = []
models = {}
# load_all_model()


def init(graphs, logged_times: list, gpu: int):
    global job_num
    global global_tensor_access
    global tensor_access_by_tensor
    global total_memory
    global handle
    global jobs_weights
    global global_graphs
    global global_tensors
    global swap_scheduler
    global parameters
    global_graphs = graphs
    jobs_weights = [weight for _ in range(len(graphs))]
    tensor_access_by_tensor = [[] for _ in range(job_num)]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # 获取当前剩余显存总量
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu)
    total_memory = nvmlDeviceGetMemoryInfo(handle).free / 1000000
    job_num = len(graphs)
    tmp = [get_framework_info(graphs[i], logged_times[i], i) for i in range(job_num)]
    global_tensor_access = [tmp[i][0] for i in range(job_num)]
    swap_scheduler = [tmp[i][1] for i in range(job_num)]
    parameters = [tmp[i][2] for i in range(job_num)]


def add_job(graph, job_id, gpu: int):
    global global_graphs
    assert job_id == len(global_graphs) or global_graphs[job_id] is None
    if job_id == len(global_graphs):
        global_graphs.append(graph)
    else:
        global_graphs[job_id] = graph
    init(global_graphs, [[] for _ in range(job_num)], gpu)


def remove_job(job_id, gpu: int):
    global global_graphs
    global_graphs[job_id] = None
    init(global_graphs, [], gpu)


def generate_scheduling_plan(logged_times, gpu: int):
    # 如果是此时logged_times已经清空，则
    # logged_times: [[(operation_id, [time, time, time])]]，外层索引为job_id
    global total_memory
    global global_tensors
    init(global_graphs, logged_times, gpu)
    # 指数加权平均更新估计时间
    tensor_nums = list(map(lambda x: len(x), tensor_access_by_tensor))
    swap_out_number_limits = [int(weight * tensor_num) for weight, tensor_num in zip(jobs_weights, tensor_nums)]
    swap_out_number = [0 for _ in tensor_nums]
    swapped_out_tensor = set()
    swap_out_dict = {}
    swapped_in_access = set()
    recomputations = []
    recomputation_tensor = set()
    # 上一轮没有成功的swap_out时为False
    swapped_flag = True
    recomputation_flag = True
    iter = 0
    original_memory_used = 0
    original_memory_footprint = None
    last_memory_used = 0
    max_memory = 0
    job_id_ordered_by_weights = list(map(lambda x: x[0], sorted([(job_id, weights) for job_id, weights in enumerate(jobs_weights)], key=lambda x: x[1], reverse=True)))
    while swapped_flag or (recomputation_flag and enable_recomputation):
        # MB
        total_memory = nvmlDeviceGetMemoryInfo(handle).free / 1000000
        max_memory, max_tensors, last_input_accesses, max_time, foot_prints, time_axis = run_global_memory_analysis(global_tensor_access, swap_scheduler, swapped_out_tensor, recomputation_tensor,
                                                                                                                    tensor_access_by_tensor)
        if iter == 0:
            original_memory_used = max_memory
            original_memory_footprint = foot_prints
            liveness_analysis(global_tensor_access)
        else:
            last_memory_used = max_memory
        print(f'iter:{iter}, max_memory:{max_memory}')
        max_tensors = sorted(max_tensors, key=lambda x: x.size, reverse=True)
        if swapped_flag:
            swapped_flag = False
            for tensor in max_tensors:
                # 对该张量进行swap_out计划的安排
                is_new_parameter = tensor.is_parameter and tensor_access_by_tensor[tensor.job_id][tensor][0].operation_name != 'feed_dict' and len(tensor_access_by_tensor[tensor.job_id][tensor]) == 1
                if not is_new_parameter:
                    if swap_out_number[tensor.job_id] <= swap_out_number_limits[tensor.job_id] and len(tensor_access_by_tensor[tensor.job_id][tensor]) > 1:
                        # swapped_out表示所有可能的swap_in已经调度过了
                        if tensor not in swapped_out_tensor:
                            all_access_of_tensor = tensor_access_by_tensor[tensor.job_id][tensor][1:]
                            # 首先确定swap_out的时间范围，最迟不能超过此时此刻，最早不能超过第一次访问结束时刻
                            output_access = tensor_access_by_tensor[tensor.job_id][tensor][0]
                            assert output_access.access_type == AccessType.output
                            if last_input_accesses[tensor.job_id] is not None:
                                # 此时此刻
                                back_boundary = last_input_accesses[tensor.job_id].time
                            else:
                                last_time_access = tensor_access_by_tensor[tensor.job_id][tensor][-1]
                                back_boundary = last_time_access.time + tensor.swap_time
                            succeed = False
                            front_boundary = output_access.time
                            failed_input_access = []
                            swap_out_succeed = True
                            have_next_ITA = True
                            # 如果是因为swap out放不下，则不用继续更新可行区间了，直接break
                            while not succeed and front_boundary < back_boundary and swap_out_succeed and have_next_ITA:
                                swap_out_task = SwapTask(tensor, output_access.time, tensor.swap_time, TaskType.swap_out, front_boundary=front_boundary, back_boundary=back_boundary)
                                free_intervals = get_free_intervals(swap_out_task, swap_scheduler[swap_out_task.tensor.job_id])
                                selected_first_access_index = None
                                # 选出能容纳该任务的剩余空间
                                swap_out_succeed = False
                                have_next_ITA = False
                                for interval in free_intervals:
                                    if interval[1] - interval[0] >= swap_out_task.time_cost:
                                        swap_out_succeed = True
                                        swap_out_task.start_time = interval[0]
                                        swap_out_task.end_time = swap_out_task.start_time + swap_out_task.time_cost
                                        swap_scheduler[swap_out_task.tensor.job_id].append(swap_out_task)
                                        # 看一下后面第一个swap_in能否放下
                                        for i, access in enumerate(all_access_of_tensor):
                                            # 找到后面第一个访问
                                            if access.start_time >= swap_out_task.start_time and access not in failed_input_access:
                                                have_next_ITA = True
                                                if can_next_input_access_swap_in(i, all_access_of_tensor, swap_out_task, swap_scheduler):
                                                    swapped_out_tensor.add(tensor)
                                                    swap_out_dict[tensor] = swap_out_task
                                                    swapped_in_access.add(access)
                                                    swap_out_number[tensor.job_id] += 1
                                                    selected_first_access_index = i
                                                    succeed = True
                                                    swapped_flag = True
                                                else:
                                                    failed_input_access.append(access)
                                                    swap_scheduler[swap_out_task.tensor.job_id].remove(swap_out_task)
                                                    # 修正swap_out_task前向限制为这个失败的input_access的结束时间
                                                    front_boundary = access.end_time
                                                    assert tensor not in swapped_out_tensor
                                                    # swapped_out_tensor.remove(tensor)
                                                break
                                        if not succeed:
                                            if swap_out_task in swap_scheduler[swap_out_task.tensor.job_id]:
                                                swap_scheduler[swap_out_task.tensor.job_id].remove(swap_out_task)
                                                # 如果不是因为swap out没安排下则重新生成区间
                                                break
                                        else:
                                            break
                            # 安排失败
                            if not succeed:
                                continue
                            if not is_new_parameter:
                                # 后续的能换入的话，则把前一个的release_flag设为True
                                for i in range(selected_first_access_index + 1, len(all_access_of_tensor)):
                                    access = all_access_of_tensor[i]
                                    if i == 0 or access in swapped_in_access:
                                        continue
                                    else:
                                        if can_next_input_access_swap_in(i, all_access_of_tensor, swap_out_task, swap_scheduler):
                                            # print(f'成功{access}')
                                            swapped_in_access.add(access)
                                            if all_access_of_tensor[i - 1].start_time > swap_out_task.end_time:
                                                all_access_of_tensor[i - 1].release_flag = True
                            if swapped_flag:
                                break
                # 如果是新参数，则尝试对新参数进行swap out，对对应的旧参数进行swap in
                else:
                    output_access = tensor_access_by_tensor[tensor.job_id][tensor][0]
                    assert output_access.access_type == AccessType.output
                    # TODO: 框架在开始下一个batch的计算前需要等待最后一个swap结束
                    swap_out_task = SwapTask(tensor, time=output_access.time, time_cost=tensor.swap_time, task_type=TaskType.swap_out, front_boundary=output_access.end_time, back_boundary=float('inf'))
                    free_intervals = get_free_intervals(swap_out_task, swap_scheduler[swap_out_task.tensor.job_id])
                    for interval in free_intervals:
                        if interval[1] - interval[0] >= swap_out_task.time_cost:
                            swap_out_task.start_time = interval[0]
                            swap_out_task.end_time = swap_out_task.start_time + swap_out_task.time_cost
                            swap_scheduler[swap_out_task.tensor.job_id].append(swap_out_task)
                            # 找到对应的旧参数张量
                            # 由于二者可行域无关，所以直接查看对应的swap in 能否调度
                            for t in tensor.source_tensors:
                                if t.is_parameter:
                                    # 试图swap in
                                    # 找到第一次访问
                                    first_access = tensor_access_by_tensor[t.job_id][t][0]
                                    assert first_access.access_type == AccessType.output and first_access.operator_name == 'feed_dict'
                                    swap_in_task = SwapTask(t, first_access.time, first_access.tensor.swap_time, TaskType.swap_in, front_boundary=float('-inf'), back_boundary=first_access.start_time)
                                    res = try_swap_in(swap_in_task, swap_scheduler)
                                    # assert not res, f'swap in parameter:{t} failed'
                                    if res:
                                        swapped_out_tensor.add(tensor)
                                        swap_out_dict[tensor] = swap_out_task
                                        swapped_in_access.add(first_access)
                                        swap_out_number[tensor.job_id] += 1
                                        swapped_flag = True
                                    else:
                                        swap_scheduler[swap_out_task.tensor.job_id].remove(swap_out_task)
                                        # 修正swap_out_task前向限制为这个失败的input_access的结束时间
                                        assert tensor not in swapped_out_tensor
                                    break
                            break
        elif enable_recomputation:
            recomputation_flag = False
            # 需要重计算
            if max_memory >= total_memory:
                succeed = False
                for job_id in job_id_ordered_by_weights:
                    max_tensors_filtered = []
                    for tensor in max_tensors:
                        # 张量不是参数，没被逐出过，且他的所有源张量从未被swap或recomputation
                        if not tensor.is_parameter and tensor not in swapped_out_tensor and tensor.source_tensors is not None and len(tensor.source_tensors) > 0 and \
                                False not in [t not in swapped_out_tensor for t in tensor.source_tensors] and False not in [t not in recomputations for t in tensor.source_tensors]:
                            max_tensors_filtered.append(tensor)
                    if len(max_tensors_filtered) == 0:
                        continue
                    max_tensors_by_metric = sorted(max_tensors_filtered, key=lambda x: x.recomputation_metric, reverse=True)
                    # 选取metric最大的张量
                    tensor = max_tensors_by_metric[0]
                    # 找到此刻对应的下一个访问
                    now_time = max_time[job_id]
                    all_access_of_tensor = tensor_access_by_tensor[tensor.job_id][tensor]
                    for i, access in enumerate(all_access_of_tensor):
                        if access.access_type == AccessType.input and access not in recomputations:
                            if access.start_time >= now_time:
                                recomputations.append(access)
                                all_access_of_tensor[i - 1].release_flag = True
                                recomputation_flag = True
                                recomputation_tensor.add(access.tensor)
                                # 无需插入重计算导致的张量访问，因为一个不需要swap in 的input访问不会改变显存占用
                                # print('recompute')
                                break
        iter += 1
    fig = go.Figure(data=[go.Scatter(x=list(original_memory_footprint[0].keys()), y=list(original_memory_footprint[0].values())), go.Scatter(x=list(foot_prints[0].keys()), y=list(foot_prints[0].values()))])
    plotly.offline.plot(fig, filename='../../pic/footprint.html')
    total_memory = nvmlDeviceGetMemoryInfo(handle).free / 1000000
    stats = 'succeed' if max_memory < total_memory else ' failure'
    print(f'scheduling {stats}')
    draw_all_task(tensor_access_by_tensor, swap_scheduler, job_num)
    memory_saved_ratio = format((1 - last_memory_used / original_memory_used) * 100, '.2f')
    print(f'memory_saved_ratio:{memory_saved_ratio}%')
    print(f'swap ratio:{len(swap_scheduler[0]) / len(global_tensors)}')
    print(f'recomputations:{recomputations}')
    return generate_swap_recomputation_release_order(tensor_access_by_tensor, swap_scheduler, recomputations, job_num)


def multiprocess_init(global_message_queue: multiprocessing.Queue, global_control_queue: multiprocessing.Queue):
    pass
    # logged_times = []
    # log_repeat = 0
    # while True:
    #     if not global_message_queue.empty():
    #         global_message = global_message_queue.get()
    #         job_id = global_message[0]
    #         message_type = global_message[1][0]
    #         message_graph = global_message[1][1]
    #
    #         if message_type == 0:
    #             # todo add to add_job
    #             global job_num
    #             job_num += 1
    #             logged_times.append([])
    #             global_graphs.append(message_graph)
    #             tensor_num = len(message_graph)
    #             for i in range(tensor_num):
    #                 logged_times[job_id].append([50])
    #             # logged_times[job_id] = [[50, 0.01], [50, 0.01], [50, 351], [50, 0.01], [50, 87], [50, 136], [50, 98], [50, 0.01], [50, 77], [50, 0.01], [50, 23], [50, 85], [50, 33], [50, 0.01], [50, 63], [50, 0.01], [50, 23],
    #             #      [50, 71], [50, 0.01], [50, 80], [50, 65], [50, 56], [50, 69], [50, 56], [50, 203], [50, 28], [50, 66], [50, 60], [50, 66], [50, 29], [50, 75], [50, 62], [50, 32], [50, 24], [50, 81],
    #             #      [50, 114], [50, 50], [50, 42], [50, 707], [50, 554], [50, 121]]
    #             s = time.time()
    #             release_order, swap_order, recomputation_order = generate_scheduling_plan(logged_times, 0)
    #             print(f'time:{time.time() - s}')
    #             control_messages = []
    #             for i in range(job_num):
    #                 control_message = [swap_order[i], release_order[i], recomputation_order[i]]
    #                 control_messages.append(control_message)
    #                 # global_control_queue.put(control_messages)
    #         else:
    #             for node_message in message_graph:
    #                 logged_times[job_id][node_message[0]].append(node_message[1])
    #             log_repeat += 1
    #             if log_repeat == 10:
    #                 release_order, swap_order, recomputation_order = generate_scheduling_plan(logged_times, 0)
    #                 control_messages = []
    #                 for i in range(job_num):
    #                     print(swap_order)
    #                     control_message = [swap_order[i], release_order[i], recomputation_order[i]]
    #                     control_messages.append(control_message)
    #                 global_control_queue.put(control_messages)
    #             # print(logged_times[0])
