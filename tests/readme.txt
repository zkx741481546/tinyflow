实验脚本在test/Experiment路径下，确保每次运行Experiment路径下有log文件夹

type是调度方式的选择, 0.不调度 1.capuchin 2.vdnn
如果内存超限可以跳过type0(不调度的方法)

1. 5个神经网络单独运行，实验脚本在lab1.py内
batch_size大小修改下面语句的batch_size参数
每个网络迭代次数修改下面语句的num_setp参数
net_run(num_step=10, net_id=net_id, type=type, batch_size=16)

3. 5个神经网络随机抽取i同时启动，实验脚本在lab3.py内
num_net 随机选取的网络个数
net_id 随机选取网络种类 0:vgg16, 1:inceptionv3, 2:inceptionv4, 3:resNet, 4:denseNet
nets 同时启动的网络的net_id
batch_size和迭代次数修改同上

4. 对VGG和ResNet在batch_size32, 128, 256情况单独运行，实验脚本在lab4.py内

每个实验脚本记录都在log下, 运行前先清空log, record_i对应实验设计里记录的第i条
lab3抽取的相同神经网络用net_order区分
lab4相同神经网络不同batch_size用batch_size=32/128/256区分
