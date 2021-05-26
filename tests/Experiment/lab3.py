from tests.Experiment import VGG16_test, ResNet50_test, DenseNet_test, InceptionV3_test, InceptionV4_test
import random, time

def net_run_thread(num_step, net_id, type, batch_size, file_name = ""):
    if net_id == 0:
        vgg16 = VGG16_test.VGG16(num_step=num_step, type=type, batch_size=batch_size, gpu_num=1, file_name=file_name, need_tosave=3568.875*0.1688*1e6)
        vgg16.start()
    elif net_id==1:
        inceptionv3 = InceptionV3_test.Inceptionv3(num_step=num_step, type=type, batch_size=batch_size, gpu_num=1, file_name=file_name)
        inceptionv3.start()
    elif net_id==2:
        inceptionv4 = InceptionV4_test.Inceptionv4(num_step=num_step, type=type, batch_size=batch_size, gpu_num=1, file_name=file_name)
        inceptionv4.start()
    elif net_id == 3:
        resNet = ResNet50_test.ResNet50(num_step=num_step, type=type, batch_size=batch_size, gpu_num=1, file_name=file_name)
        resNet.start()
    elif net_id == 4:
        denseNet = DenseNet_test.DenseNet121(num_step=num_step, type=type, batch_size=batch_size, gpu_num=1, file_name=file_name)
        denseNet.start()
    return vgg16


def Experiment3():
    print("Experiment3 start")
    # num_net = random.randint(2, 5) #随机选取网络个数
    # print("随机选取", num_net, "个网络")
    nets = []
    # for i in range(num_net):
    #     net_id = random.randint(0, 4) #net_id随机选取网络种类 0:vgg16, 1:inceptionv3, 2:inceptionv4, 3:resNet, 4:denseNet
    #     nets.append(net_id)

    nets.append(0)
    nets.append(0)
    print("选取的网络", nets)
    # for type in range(3): #type是调度方式的选择, 0.不调度，1.capuchin 2.vdnn
    #     # if type == 0:
    #     #     continue
    #     i = 0
    #     for net_id in nets:
    #         i = i + 1
    #         net_run_thread(num_step=10, net_id=net_id, type=type, batch_size=16, file_name="net_order=" + str(i))
    #     time.sleep(100)


    for type in range(3):
        thread_pool = []
        for i, net_id in enumerate(nets):
            thread_pool.append(net_run_thread(num_step=10, net_id=net_id, type=type, batch_size=2, file_name="net_order=" + str(i)))
        for th in thread_pool:
            th.join()

    print("Experiment3 finish")

Experiment3()




