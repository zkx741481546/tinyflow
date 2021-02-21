import copy
from enum import Enum
import multiprocessing
import numpy as np
from functools import cmp_to_key
import plotly as py
import plotly.figure_factory as ff
from collections import defaultdict
import os
from pynvml import *

pyplt = py.offline.plot


class TaskType(Enum):
    swap_out = 0
    swap_in = 1


class AccessType(Enum):
    output = 0
    input = 1


class Tensor:
    def __init__(self, tensor_id, job_id, size, source_tensors=None):
        self.tensor_id = tensor_id
        self.job_id = job_id
        self.size = size
        self.swap_time = self.size / 50 * (np.random.random() + 0.8)
        self.source_tensors = source_tensors if source_tensors is not None else []
        self.recomputation_time = np.random.randint(1, 10) * self.swap_time
        self.recomputation_metric = self.size / self.recomputation_time

    def __repr__(self):
        return f'tensor_id:{self.tensor_id}, job_id":{self.job_id}, size:{self.size}'


class TensorAccess:
    def __init__(self, tensor, time=None, access_type=None, operation_id=None):
        self.tensor = tensor
        self.access_id = None
        self.start_time = None
        self.end_time = None
        self.time = np.random.random() * time_scale if time is None else time
        tmp = np.random.normal(loc=0.5, scale=1, size=1)[0]
        tmp = 1 if tmp > 1 else tmp
        self.run_time = 0.05 if tmp <= 0.1 else tmp
        if access_type is None:
            tmp = np.random.randint(2)
            self.access_type = AccessType.output if tmp == 0 else AccessType.input
        else:
            self.access_type = access_type
        self.release_flag = False
        self.operation_id = operation_id

    def to_tuple(self):
        return (self.tensor.tensor_id, self.time)

    def __repr__(self):
        return f'id={self.tensor.tensor_id}, time={self.time}'


class SwapTask(object):
    '''Date weighted interval'''

    def __init__(self, tensor, weight, time, time_cost, task_type: TaskType, front_boundary=None, back_boundary=None):
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
        if task_type == TaskType.swap_out:
            self.start_time = self.time
            self.end_time = self.start_time + self.time_cost
        else:
            self.end_time = self.time
            self.start_time = self.end_time - self.time_cost
        self.weight = weight
        self.execute_time = None
        self.execute_ref = None

    @classmethod
    def from_access(cls, access: TensorAccess, weight, task_type, front_boundary=None, back_boundary=None):
        return cls(access.tensor, weight, access.time, access.tensor.swap_time, task_type, front_boundary=front_boundary, back_boundary=back_boundary)

    def __repr__(self):
        return f'id={self.tensor}, type={self.task_type}, start_time={self.start_time}, end_time={self.end_time}, weight={self.weight}'


def numpy_ewma_vectorized(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def get_predicted_execution_time(op_name, input_tensors, logged_time: list):
    input_size = 0
    for tensor in input_tensors:
        input_size += tensor.size
    # TODO
    predicted_time = 0
    if len(logged_time)>0:
        predicted_time = [predicted_time]
        predicted_time.extend(logged_time)
        predicted_time = numpy_ewma_vectorized(np.array(predicted_time), 3)
    return predicted_time


def liveness_analysis(tensor_access_list):
    # 活跃性分析结果生成
    tmp = set()
    for i in range(len(tensor_access_list) - 1, -1, -1):
        tensor_access = tensor_access_list[i]
        if tensor_access.tensor not in tmp:
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


def generate_data(job_id):
    tensor_access_list = []
    swap_times = {}
    tensor_list = []
    for tensor_id in range(tensors):
        tmp = int(np.random.normal(loc=20, scale=10, size=1)[0])
        tensor_list.append(Tensor(tensor_id, job_id, tmp if tmp > 5 else 5))
    for _ in range(times):
        t = TensorAccess(tensor_list[np.random.randint(0, len(tensor_list))])
        if t.tensor in swap_times.keys():
            t.swap_time = swap_times[t.tensor]
            t.access_type = AccessType.input
        else:
            swap_times[t.tensor] = t.tensor.swap_time
            t.access_type = AccessType.output
        # 根据访问类型重新递推time，这里的time定义根据访问类型的不同而不同
        if len(tensor_access_list) > 0:
            if t.access_type == AccessType.output:
                if tensor_access_list[-1].access_type == AccessType.input:
                    t.time = tensor_access_list[-1].time + tensor_access_list[-1].run_time + t.run_time
                else:
                    t.time = tensor_access_list[-1].time + t.run_time
            else:
                if tensor_access_list[-1].access_type == AccessType.input:
                    t.time = tensor_access_list[-1].time + tensor_access_list[-1].run_time
                else:
                    t.time = tensor_access_list[-1].time
        if t.access_type == AccessType.output:
            t.end_time = t.time
            t.start_time = t.time - t.run_time
        else:
            t.start_time = t.time
            t.end_time = t.time + t.run_time

            # t.access_type = AccessType.output
        tensor_access_list.append(t)
    tensor_access_list = sorted(tensor_access_list, key=lambda x: x.time)
    for index, tensor_access in enumerate(tensor_access_list):
        tensor_access.access_id = index
    for tensor in tensor_list:
        tmp = np.random.choice(tensor_list, np.random.randint(len(tensor_access_list)))
        tensor.source_tensors = [i for i in tmp if i != tensor]
    liveness_analysis(tensor_access_list)
    print(list(map(lambda x: x.to_tuple(), tensor_access_list)))
    print(list(map(lambda x: x.tensor, tensor_access_list)))
    return tensor_access_list


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


def get_max_memory_used(tensor_access_list, swap_tasks, swapped_out_tensor, recomputation_tensor):
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
    memory_used = 0
    max_memory = float('-inf')
    in_gpu_tensors = set()
    max_memory_tensors = set()
    last_input_tensor_access = None
    max_last_access = None
    wait_to_be_released = []
    max_time = None
    for time_index, event in enumerate(time_axis):
        time = event.time
        for i in range(len(wait_to_be_released) - 1, -1, -1):
            access = wait_to_be_released[i]
            # 如果此刻时间已经过了释放时间，则释放该访问的附带影响
            if time > access.end_time:
                wait_to_be_released.pop(i)
                assert not (access.tensor not in recomputation_tensor and access.tensor not in in_gpu_tensors)
                if access.tensor in in_gpu_tensors:
                    memory_used -= access.tensor.size
                    in_gpu_tensors.remove(access.tensor)
        if isinstance(event, TensorAccess):
            if event.access_type == AccessType.output:
                memory_used += event.tensor.size
                in_gpu_tensors.add(event.tensor)
            else:
                # 用完即释放的
                # input本身并不增加gpu使用，swap in增加
                # if (event.tensor in swapped_out_tensor or event.tensor in recomputation_tensor) and event.release_flag:
                if event.release_flag:
                    # memory_used += event.tensor.size
                    # in_gpu_tensors.add(event.tensor)
                    wait_to_be_released.append(event)
                else:
                    last_input_tensor_access = event
                    # in_gpu_tensors.add(event.tensor)
        elif isinstance(event, SwapTask):
            last_event = None
            for j in range(time_index - 1, -1, -1):
                if isinstance(time_axis[j], TensorAccess) and time_axis[j].end_time <= event.start_time:
                    last_event = time_axis[j]
                    break
            # for j in range(time_index-1,-1,-1):
            #     if time_axis[j].end_time == event.start_time:
            #         last_event = time_axis[j]
            #         last_time = 0
            #         break
            # if last_event is None:
            #     for j in range(time_index - 1, -1, -1):
            #         if time_axis[j].end_time<event.start_time:
            #             last_event = time_axis[j]
            #             last_time = event.start_time - time_axis[j].end_time
            #             break
            assert last_event is not None
            event.execute_ref = last_event
            event.execute_time = event.start_time - last_event.end_time
            if event.task_type == TaskType.swap_in:
                memory_used += event.tensor.size
                in_gpu_tensors.add(event.tensor)
            else:
                memory_used -= event.tensor.size
                in_gpu_tensors.remove(event.tensor)
        if memory_used > max_memory:
            max_memory = memory_used
            max_memory_tensors = copy.copy(in_gpu_tensors)
            max_last_access = copy.copy(last_input_tensor_access)
            max_time = time
    return max_memory, max_memory_tensors, max_last_access, max_time


def run_global_memory_analysis(global_tensor_access, swap_tasks, swapped_out_tensor, recomputation_tensor):
    max_memory = 0
    max_memory_tensors = []
    last_input_accesses = []
    max_time = []
    for job_id, tensor_accesses in enumerate(global_tensor_access):
        job_max_memory, job_max_memory_tensors, last_input_access, now_time = get_max_memory_used(tensor_accesses, swap_tasks[job_id], swapped_out_tensor, recomputation_tensor)
        max_memory_tensors.extend(job_max_memory_tensors)
        last_input_accesses.append(last_input_access)
        max_time.append(now_time)
        max_memory += job_max_memory
    return max_memory, max_memory_tensors, last_input_accesses, max_time


def draw(tensor_access_list, swap_schedule):
    df = []
    id_color = {-1: 'rgb(253,15,21)', -2: 'rgb(30,30,208)', 0: 'rgb(255,128,0)', 1: 'rgb(0,255,64)'}
    for tensor_access in tensor_access_list:
        # input 蓝色，output红色
        df.append(dict(Task=f'tensor_id:{tensor_access.tensor.tensor_id}', Start=tensor_access.start_time, Finish=tensor_access.end_time,
                       Resource=-1 if tensor_access.access_type == AccessType.output else -2))
    for task in swap_schedule:
        df.append(dict(Task=f'tensor_id:{task.tensor.tensor_id}', Start=task.start_time, Finish=task.end_time, Resource=0 if task.task_type == TaskType.swap_in else 1))

    fig = ff.create_gantt(df, colors=id_color, index_col='Resource', group_tasks=True, show_colorbar=True, showgrid_x=True, showgrid_y=True, title=f'ratio={ratio}')
    fig['layout']['xaxis'].update({'type': None})
    pyplt(fig, filename=f'job{tensor_access_list[0].tensor.job_id}.html', auto_open=True)


def can_next_input_access_swap_in(i, all_access_of_tensor, swap_out_task, swap_scheduler):
    # 至少将第一个访问swap in才算成功，后续的能换入的话，则把前一个的release_flag设为True
    access = all_access_of_tensor[i]
    swap_in_task = SwapTask(access.tensor, None, access.time, access.tensor.swap_time, TaskType.swap_in,
                            front_boundary=swap_out_task.end_time if swap_out_task.end_time > all_access_of_tensor[i - 1].end_time else all_access_of_tensor[i - 1].end_time,
                            back_boundary=access.time)
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


def get_framework_info(info, logged_time, job_id):
    tensors = {}
    tensor_access_list = []
    global_time = 0
    #   operation_id
    for output_tensor_id, input_tensor_id, output_tensor_size, operation_name in enumerate(info):
        input_tensors = []
        for tensor_id in input_tensor_id:
            input_tensor = tensors[tensor_id]
            input_tensors.append(input_tensor)
            input_access = TensorAccess(tensor=input_tensor, time=global_time, access_type=AccessType.input, operation_id=output_tensor_id)
            tensor_access_list.append(input_access)
        output_tensor = Tensor(tensor_id=output_tensor_id, job_id=job_id, size=output_tensor_size, source_tensors=input_tensors)
        time_cost = get_predicted_execution_time(operation_name, input_tensors, logged_time[output_tensor_id])
        global_time += time_cost
        output_access = TensorAccess(tensor=output_tensor, time=time_cost, access_type=AccessType.output, operation_id=output_tensor_id)
        tensor_access_list.append(output_access)
        tensors[output_tensor.tensor_id] = output_access
    # tensors = list(tensors.values())
    tensor_access_list = sorted(tensor_access_list, key=lambda x: x.time)
    liveness_analysis(tensor_access_list)
    return tensor_access_list


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
handle = None
enable_recomputation = True
global_graphs = []


def init(graphs, logged_times: list, gpu: int):
    global job_num
    global global_tensor_access
    global tensor_access_by_tensor
    global total_memory
    global handle
    global jobs_weights
    global global_graphs
    global_graphs = graphs
    jobs_weights = [weight for _ in range(len(graphs))]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # 获取当前剩余显存总量
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu)
    total_memory = nvmlDeviceGetMemoryInfo(handle).free

    job_num = len(graphs)
    global_tensor_access = [get_framework_info(graphs[i], logged_times[i], i) for i in range(job_num)]
    # task-tensor-access
    # [dict()]，外层索引为job_id, 内层为tensor对象
    tensor_access_by_tensor = []
    for i in range(job_num):
        tensor_accesses = global_tensor_access[i]
        dic = defaultdict(list)
        for access in tensor_accesses:
            dic[access.tensor].append(access)
        for k, v in dic.items():
            dic[k] = sorted(v, key=lambda x: x.time)
        tensor_access_by_tensor.append(dic)


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
    init(global_graphs, logged_times, gpu)
    # 指数加权平均更新估计时间
    tensor_nums = list(map(lambda x: len(x), tensor_access_by_tensor))
    swap_out_number_limits = [int(weight * tensor_num) for weight, tensor_num in zip(jobs_weights, tensor_nums)]
    swap_out_number = [0 for _ in tensor_nums]
    swap_scheduler = [[] for _ in range(job_num)]
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
    last_memory_used = 0
    max_memory = 0
    job_id_ordered_by_weights = list(map(lambda x: x[0], sorted([(job_id, weights) for job_id, weights in enumerate(jobs_weights)], key=lambda x: x[1], reverse=True)))
    while swapped_flag or (recomputation_flag and enable_recomputation):
        total_memory = nvmlDeviceGetMemoryInfo(handle).free
        max_memory, max_tensors, last_input_accesses, max_time = run_global_memory_analysis(global_tensor_access, swap_scheduler, swapped_out_tensor, recomputation_tensor)
        if iter == 0:
            original_memory_used = max_memory
        else:
            last_memory_used = max_memory
        print(f'iter:{iter}, max_memory:{max_memory}')
        max_tensors = sorted(max_tensors, key=lambda x: x.size, reverse=True)
        if swapped_flag:
            swapped_flag = False
            for tensor in max_tensors:
                # 对该张量进行swap_out计划的安排
                if swap_out_number[tensor.job_id] <= swap_out_number_limits[tensor.job_id] and len(tensor_access_by_tensor[tensor.job_id][tensor]) > 1:
                    # swapped_out表示所有可能的swap_in已经调度过了
                    if tensor not in swapped_out_tensor:
                        all_access_of_tensor = tensor_access_by_tensor[tensor.job_id][tensor]
                        # 首先确定swap_out的时间范围，最迟不能超过此时此刻，最早不能超过第一次访问结束时刻
                        output_access = tensor_access_by_tensor[tensor.job_id][tensor][0]
                        assert output_access.access_type == AccessType.output
                        if last_input_accesses[tensor.job_id] is not None:
                            # 此时此刻
                            back_boundary = last_input_accesses[tensor.job_id].time
                        else:
                            last_time_access = tensor_access_by_tensor[tensor.job_id][tensor][-1]
                            back_boundary = last_time_access.time + tensor.swap_time
                        swap_out_task = SwapTask(tensor, None, output_access.time, tensor.swap_time, TaskType.swap_out, front_boundary=output_access.time, back_boundary=back_boundary)
                        free_intervals = get_free_intervals(swap_out_task, swap_scheduler[swap_out_task.tensor.job_id])
                        succeed = False
                        selected_first_access_index = None
                        # 选出能容纳该任务的剩余空间
                        for interval in free_intervals:
                            if interval[1] - interval[0] >= swap_out_task.time_cost:
                                swap_out_task.start_time = interval[0]
                                swap_out_task.end_time = swap_out_task.start_time + swap_out_task.time_cost
                                swap_scheduler[swap_out_task.tensor.job_id].append(swap_out_task)
                                # 看一下后面第一个swap_in能否放下
                                first_access_index = None
                                for i, access in enumerate(all_access_of_tensor):
                                    if access.start_time > swap_out_task.end_time:
                                        first_access_index = i
                                        break
                                if first_access_index is not None and can_next_input_access_swap_in(first_access_index, all_access_of_tensor, swap_out_task, swap_scheduler):
                                    access = all_access_of_tensor[first_access_index]
                                    swapped_out_tensor.add(tensor)
                                    swap_out_dict[tensor] = swap_out_task
                                    swapped_in_access.add(access)
                                    swap_out_number[tensor.job_id] += 1
                                    selected_first_access_index = first_access_index
                                    succeed = True
                                    swapped_flag = True
                                    break
                                else:
                                    swap_scheduler[swap_out_task.tensor.job_id].remove(swap_out_task)
                                    continue
                        # 安排失败
                        if not succeed:
                            continue
                        # 至少将第一个访问swap in才算成功，后续的能换入的话，则把前一个的release_flag设为True
                        for i in range(selected_first_access_index + 1, len(all_access_of_tensor)):
                            access = all_access_of_tensor[i]
                            if i == 0 or access in swapped_in_access:
                                continue
                            else:
                                if can_next_input_access_swap_in(i, all_access_of_tensor, swap_out_task, swap_scheduler):
                                    # print(f'成功{access}')
                                    swapped_out_tensor.add(tensor)
                                    swap_out_dict[tensor] = swap_out_task
                                    swapped_in_access.add(access)
                                    if all_access_of_tensor[i - 1].start_time > swap_out_task.end_time:
                                        all_access_of_tensor[i - 1].release_flag = True
                        if swapped_flag:
                            break
        elif enable_recomputation:
            recomputation_flag = False
            # 需要重计算
            if max_memory >= total_memory:
                succeed = False
                for job_id in job_id_ordered_by_weights:
                    max_tensors_filtered = []
                    for tensor in max_tensors:
                        # 张量没被逐出过，且他的所有源张量从未被swap或recomputation
                        if tensor not in swapped_out_tensor and tensor.source_tensors is not None and len(tensor.source_tensors) > 0 and \
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
    stats = 'succeed' if max_memory < total_memory else ' failure'
    print(f'scheduling {stats}')
    # draw_all_task(tensor_access_by_tensor, swap_scheduler, job_num)
    memory_saved_ratio = format((1 - last_memory_used / original_memory_used) * 100, '.2f')
    print(f'memory_saved_ratio:{memory_saved_ratio}%')
    return generate_swap_recomputation_release_order(tensor_access_by_tensor, swap_scheduler, recomputations, job_num)


def multiprocess_init(global_message_queue: multiprocessing.Queue, global_control_queue: multiprocessing.Queue):
    while True:
        if not global_message_queue.empty():
            global_message = global_message_queue.get()
            job_id = global_message[0]
            message_type = global_message[1][0]
            message_graph = global_message[1][1]

            # todo add to add_job
            global job_num
            job_num += 1

            add_job(message_graph, job_id, 0)

