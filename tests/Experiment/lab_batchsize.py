import os
import sys
sys.path.append('../../')
from tests.Experiment import VGG16_test, ResNet50_test, DenseNet_test, InceptionV3_test, InceptionV4_test
import random, time

from tests.Experiment.log.result import get_result

gpu = 1


def generate_job(num_step, net_id, type, batch_size, path, need_tosave, file_name=""):
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


def Experiment2():
    net_names = ['VGG', 'InceptionV3', 'InceptionV4', 'ResNet', 'DenseNet']
    budget = {
        'VGG': {
            2: 2494.0,
            4: 2729.33,
            8: 3229.33,
            16: 4076.67,
            32: 6478.67
        },
        'Inceptionv3': {
            2: 1062.0,
            4: 1145.33,
            8: 1444.67,
            16: 2145.33,
            32: 3441.33,
        },
        'Inceptionv4': {
            2: 1398.0,
            4: 1353.33,
            8: 2060.0,
            16: 3186.67,
        },
        'ResNet': {
            2: 1617.33,
            4: 1240.0,
            8: 1315.33,
            16: 1617.33,
            32: 2849.33
        },
        'DenseNet': {
            2: 1140.67,
            4: 1164.0,
            8: 1280.67,
            16: 2222.00,
        }
    }
    for net_id in range(5):
        repeat_times = 3
        print("Experiment1 start")
        net_name = net_names[net_id]
        num_net = 1
        for i, batch_size in enumerate(budget[net_name].keys()):
            path = f'./log/{net_name}_bs{batch_size}/'
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
            nets = []
            for _ in range(num_net):
                # net_id = random.randint(0, 4) #net_id随机选取网络种类 0:vgg16, 1:inceptionv3, 2:inceptionv4, 3:resNet, 4:denseNet
                nets.append(net_id)
            # print("选取的网络", nets)
            for t in range(repeat_times):
                print(f'repeat_times:{t}')
                for type in range(3):  # type是调度方式的选择, 0.不调度，1.capuchin 2.vdnn
                    print(f'net: {list(map(lambda x: net_names[x], nets))}, batchsize:{batch_size}')
                    job_pool = [generate_job(num_step=50, net_id=net_id, type=type, batch_size=batch_size, path=path, file_name=f"_repeat_time={t}_net_order={i}", need_tosave=budget[net_name][batch_size]) for i, net_id in enumerate(nets)]
                    for job in job_pool:
                        job.start()
                    for job in job_pool:
                        job.join()
            get_result(path, repeat_times=repeat_times)
            print("Experiment2 finish")


Experiment2()
