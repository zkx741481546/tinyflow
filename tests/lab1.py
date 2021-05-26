from tests.Experiment import VGG16_test, ResNet50_test, DenseNet_test, InceptionV3_test, InceptionV4_test
def net_run(num_step, net_id, type, batch_size):
    if net_id == 0:
        vgg16 = VGG16_test.VGG16(num_step=num_step, type=type, batch_size=batch_size, gpu_num=0, file_name='')
        vgg16.run()
    elif net_id==1:
        inceptionV3 = InceptionV3_test.Inceptionv3(num_step=num_step, type=type, batch_size=batch_size, gpu_num=0, file_name='')
        inceptionV3.run()
    elif net_id==2:
        inceptionV4 = InceptionV4_test.Inceptionv4(num_step=num_step, type=type, batch_size=batch_size, gpu_num=0, file_name='')
        inceptionV4.run()
    elif net_id == 3:
        resNet50 = ResNet50_test.ResNet50(num_step=num_step, type=type, batch_size=batch_size, gpu_num=0, file_name='')
        resNet50.run()
    elif net_id == 4:
        denseNet = DenseNet_test.DenseNet121(num_step=num_step, type=type, batch_size=batch_size, gpu_num=0, file_name='')
        denseNet.run()

def Experiment1():
    print("Experiment1 start")
    for net_id in range(5):  #net_id是不同网络, 0.VGG16，1.Inceptionv3 2.Inceptionv4 3.ResNet50 4.DenseNet
        for type in range(3): #type是调度方式的选择, 0.不调度，1.capuchin 2.vdnn 3.自己的调度
            # if type == 0 or type==2:
            #     continue
            print("net_id:", net_id, "type:", type)
            net_run(num_step=10, net_id=net_id, type=type, batch_size=10)
    print("Experiment1 finish")

Experiment1()