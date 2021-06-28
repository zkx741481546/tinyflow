import os
import sys
sys.path.append('../../')
from tests.Experiment import VGG16_test, ResNet50_test, DenseNet_test, InceptionV3_test, InceptionV4_test
import random, time

from tests.Experiment.log.result import get_result

gpu = 1


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


def Experiment3():
    net_names = ['VGG', 'InceptionV3', 'InceptionV4', 'ResNet', 'DenseNet']
    repeat_times = 3
    num_net = 3
    batch_size = 2
    file = open('./log/experiment3_log.txt', 'w+')
    for exp_id in range(5):
        nets = []
        for _ in range(num_net):
            net_id = random.randint(0, 4) #net_id随机选取网络种类 0:vgg16, 1:inceptionv3, 2:inceptionv4, 3:resNet, 4:denseNet
            nets.append(net_id)
        file.writelines(f'exp_id:{exp_id}, nets:{nets}')
        file.flush()
        print("Experiment3 start")
        print("选取的网络", nets)
        path = f'./log/Experiment3/{exp_id}'
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        for t in range(repeat_times):
            print(f'repeat_times:{t}')
            for type in range(3):  # type是调度方式的选择, 0.不调度，1.capuchin 2.vdnn
                job_pool = [generate_job(num_step=50, net_id=net_id, type=type, batch_size=batch_size, path=path, file_name=f"_repeat_time={t}_net_order={i}") for i, net_id in enumerate(nets)]
                for job in job_pool:
                    job.start()
                for job in job_pool:
                    job.join()
        get_result(path, repeat_times=3)
        print("Experiment3 finish")
    file.close()

Experiment3()