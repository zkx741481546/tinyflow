import os
import sys

import numpy as np
from pycode.tinyflow import ndarray

sys.path.append('../../')
from tests.Experiment import VGG16_test, ResNet50_test, DenseNet_test, InceptionV3_test, InceptionV4_test
import random, time

from tests.Experiment.log.result import get_result

gpu = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def generate_job(num_step, net_id, type, batch_size, path, need_tosave, file_name=""):
    need_tosave *= 1e6

    if net_id == 0:
        vgg16 = VGG16_test.VGG16(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu, path=path, file_name=file_name, n_class=1000, need_tosave=need_tosave)
        return vgg16
    elif net_id == 1:
        inceptionv3 = InceptionV3_test.Inceptionv3(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu, path=path, file_name=file_name, need_tosave=need_tosave)
        return inceptionv3
    elif net_id == 2:
        inceptionv4 = InceptionV4_test.Inceptionv4(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu, path=path, file_name=file_name, need_tosave=need_tosave)
        return inceptionv4
    elif net_id == 3:
        resNet = ResNet50_test.ResNet50(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu, path=path, file_name=file_name, need_tosave=need_tosave)
        return resNet
    elif net_id == 4:
        denseNet = DenseNet_test.DenseNet121(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu, path=path, file_name=file_name, need_tosave=need_tosave)
        return denseNet


def Experiment1():
    net_names = ['VGG', 'InceptionV3', 'InceptionV4', 'ResNet', 'DenseNet']
    budget = {
        'VGG': {
            1: {2: 2494.0, 16: 4076.67},
            2: {2: 4815.67},
            3: {2: 6566.0}
        },
        'InceptionV3': {
            1: {2: 1062.0, 16: 2145.33},
            2: {2: 2085.0},
            3: {2: 2757.33}
        },
        'InceptionV4': {
            1: {2: 1398.0, 16: 3186.67},
            2: {2: 2862.33},
            3: {2: 3224.67}
        },
        'ResNet': {
            1: {2: 1191.33, 16: 1617.33},
            2: {2: 2115.0},
            3: {2: 2869.33}
        },
        'DenseNet': {
            1: {2: 1140.67, 16: 2222.0, },
            2: {2: 2287.0},
            3: {2: 3426.67}
        }
    }
    for net_id in range(5):
        repeat_times = 1
        print("Experiment1 start")
        net_name = net_names[net_id]
        for i, num_net in enumerate([1, 1, 2, 3]):
            if i == 0:
                batch_size = 16
                net_name_ = net_name
            else:
                batch_size = 2
                net_name_ = net_name + f' x{i}'
            path = f'./log/{net_name_}/'
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
            nets = []
            for _ in range(num_net):
                # net_id = random.randint(0, 4) #net_id随机选取网络种类 0:vgg16, 1:inceptionv3, 2:inceptionv4, 3:resNet, 4:denseNet
                nets.append(net_id)
            print("选取的网络", list(map(lambda x: net_names[x], nets)))
            for t in range(repeat_times):
                print(f'repeat_times:{t}')
                for type in range(3):  # type是调度方式的选择, 0.不调度，1.capuchin 2.vdnn
                    if type != 1:
                        continue
                    need_tosave = 0
                    if type == 1:
                        bud = budget[net_name][num_net][batch_size]
                        # 总显存=预算+need_tosave(额外占用空间)
                        need_tosave = 11019 - bud
                        print(f'need_tosave:{need_tosave}')
                        outspace = []
                        size = need_tosave * 1e6 / 4
                        gctx = ndarray.gpu(0)
                        while size > 0:
                            if size > 10000 * 10000:
                                outspace.append(ndarray.array(np.ones((10000, 10000)) * 0.01, ctx=gctx))
                                size -= 10000 * 10000
                            else:
                                need_sqrt = int(pow(size, 0.5))
                                if need_sqrt <= 0:
                                    break
                                outspace.append(ndarray.array(np.ones((need_sqrt, need_sqrt)) * 0.01, ctx=gctx))
                                size -= need_sqrt * need_sqrt
                        print('finish extra matrix generation')
                    job_pool = [
                        generate_job(num_step=50, net_id=net_id, type=type, batch_size=batch_size, path=path, file_name=f"_repeat_time={t}_net_order={i}", need_tosave=need_tosave) for
                        i, net_id in enumerate(nets)]
                    for job in job_pool:
                        job.start()
                    for job in job_pool:
                        job.join()
                    print(len(outspace))
            print(f'get_result:{need_tosave}')
            get_result(path, repeat_times=repeat_times, need_tosave=need_tosave)
            print("Experiment1 finish")


Experiment1()
