from tests.Experiment import VGG16_test, ResNet50_test

def net_run(num_step, net_id, type, batch_size, file_name = ""):
    if net_id == 0:
        vgg16 = VGG16_test.VGG16(num_step=num_step, type=type, batch_size=batch_size, gpu_num=0, file_name=file_name)
        vgg16.run()
    elif net_id==1:
        resNet50 = ResNet50_test.ResNet50(num_step=num_step, type=type, batch_size=batch_size, gpu_num=0, file_name=file_name)
        resNet50.run()
def Experiment4():
    #type是调度方式的选择, 0.不调度，1.capuchin 2.vdnn
    print("Experiment4 start")
    # a = sys.stdin.readline()
    # num_step = int(a)
    for net_id in range(2):
        for type in range(3):
            # if type == 0:
            #     continue
            for batch_size in (32, 128, 256):
                # print(batch_size)
                print("net_id:", net_id, "type:", type, "batch_size", batch_size)
                net_run(num_step=10, net_id=net_id, type=type, batch_size=batch_size, file_name="batch_size=" + str(batch_size))
    print("Experiment4 finish")

Experiment4()