import numpy as np
from pycode.tinyflow import mainV2 as mp
from pycode.tinyflow import autodiff as ad
from pycode.tinyflow.tools import *
from pycode.tinyflow import ndarray
import threading, pynvml, multiprocessing, os, datetime, time
from multiprocessing import Process

GPU = load_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = f'{GPU}'

class ResNet50():
    def __init__(self, num_step, batch_size, gpu_num):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
        self.gpu_num = gpu_num

        self.dropout_rate = 0.5
        self.image_channel = 3
        self.num_step = num_step
        self.batch_size = batch_size
        self.ad = ad


    def identity_block(self, inputs, kernel_size, in_filter, out_filters, block_name, executor_ctx):

        f1, f2, f3 = out_filters

        W1 = self.ad.Variable(block_name + "W1")
        W2 = self.ad.Variable(block_name + "W2")
        W3 = self.ad.Variable(block_name + "W3")
        W1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f1, in_filter, 1, 1)), executor_ctx)
        W2_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f2, f1, kernel_size, kernel_size)), executor_ctx)
        W3_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f3, f2, 1, 1)), executor_ctx)

        # conv1
        conv1 = self.ad.convolution_2d_forward_op(inputs, W1, "NCHW", "SAME", 1, 1)
        bn1 = self.ad.bn_forward_op(conv1, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")

        #conv2
        conv2 = self.ad.convolution_2d_forward_op(act1, W2, "NCHW", "SAME", 1, 1)
        bn2 = self.ad.bn_forward_op(conv2, "NCHW", "pre_activation")
        act2 = self.ad.activation_forward_op(bn2, "NCHW", "relu")

        #conv3
        conv3 = self.ad.convolution_2d_forward_op(act2, W3, "NCHW", "VALID", 1, 1)
        bn3 = self.ad.bn_forward_op(conv3, "NCHW", "pre_activation")

        #shortcut
        shortcut = inputs
        add = self.ad.add_op(bn3, shortcut)
        act4 = self.ad.activation_forward_op(add, "NCHW", "relu")

        dict = {W1: W1_val, W2: W2_val, W3: W3_val}
        return act4, dict

    def convolutional_block(self, inputs, kernel_size, in_filter, out_filters, block_name, stride, executor_ctx):
        f1, f2, f3 = out_filters

        W1 = self.ad.Variable(block_name + "W1")
        W2 = self.ad.Variable(block_name + "W2")
        W3 = self.ad.Variable(block_name + "W3")
        W_shortcut = self.ad.Variable(block_name + "W_shortcut")
        W1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f1, in_filter, 1, 1)), executor_ctx)
        W2_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f2, f1, kernel_size, kernel_size)), executor_ctx)
        W3_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f3, f2, 1, 1)), executor_ctx)
        W_shortcut_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f3, in_filter, 1, 1)), executor_ctx)

        # conv1
        conv1 = self.ad.convolution_2d_forward_op(inputs, W1, "NCHW", "VALID", stride, stride)
        bn1 = self.ad.bn_forward_op(conv1, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")

        # conv2
        conv2 = self.ad.convolution_2d_forward_op(act1, W2, "NCHW", "SAME", 1, 1)
        bn2 = self.ad.bn_forward_op(conv2, "NCHW", "pre_activation")
        act2 = self.ad.activation_forward_op(bn2, "NCHW", "relu")

        # conv3
        conv3 = self.ad.convolution_2d_forward_op(act2, W3, "NCHW", "VALID", 1, 1)
        bn3 = self.ad.bn_forward_op(conv3, "NCHW", "pre_activation")

        #shortcut_path
        conv4 = self.ad.convolution_2d_forward_op(inputs, W_shortcut, "NCHW", "VALID", stride, stride)
        shortcut = self.ad.bn_forward_op(conv4, "NCHW", "pre_activation")

        # shortcut
        add = self.ad.add_op(bn3, shortcut)
        act4 = self.ad.activation_forward_op(add, "NCHW", "relu")

        dict = {W1: W1_val, W2: W2_val, W3: W3_val, W_shortcut: W_shortcut_val}
        return act4, dict

    def res_net(self, executor_ctx, top_control_queue, top_message_queue, n_class, X_val, y_val):


        X = self.ad.Placeholder("X")
        y_ = self.ad.Placeholder("y_")
        W1 = self.ad.Variable("W1")
        W6 = self.ad.Variable("W6")
        b6 = self.ad.Variable("b6")
        W7 = self.ad.Variable("W7")
        b7 = self.ad.Variable("b7")

        # zero pading
        # pad = 3   stride=1   pool_size=1*1
        pool0 = self.ad.pooling_2d_forward_op(X, "NCHW", "max", 3, 3, 1, 1, 1, 1)  # 3*224*224 ->  3*230*230

        #conv1
        conv1 = self.ad.convolution_2d_forward_op(pool0, W1, "NCHW", "VALID", 2, 2)  # stride = 2  3*230*230 -> 64*112*112
        bn1 = self.ad.bn_forward_op(conv1, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")
        # pad = 0   stride=2   pool_size=3*3
        pool1 = self.ad.pooling_2d_forward_op(act1, "NCHW", "max", 0, 0, 2, 2, 3, 3)  # 64*112*112 -> 64*112*112

        #conv2_x
        conv2, dict2 = self.convolutional_block(inputs=pool1, kernel_size=3, in_filter=64, out_filters=[64, 64, 256], block_name="2a_", stride=1, executor_ctx=executor_ctx)
        iden2_1, dict2_1 = self.identity_block(inputs=conv2, kernel_size=3, in_filter=256, out_filters=[64, 64, 256], block_name="2b_", executor_ctx=executor_ctx)
        iden2_2, dict2_2 = self.identity_block(iden2_1, 3, 256, [64, 64, 256], "2c_", executor_ctx)

        #conv3_x
        conv3, dict3 = self.convolutional_block(iden2_2, 3, 256, [128, 128, 512], "3a_", 2, executor_ctx)
        iden3_1, dict3_1 = self.identity_block(conv3, 3, 512, [128, 128, 512], "3b_", executor_ctx)
        iden3_2, dict3_2 = self.identity_block(iden3_1, 3, 512, [128, 128, 512], "3c_", executor_ctx)
        iden3_3, dict3_3 = self.identity_block(iden3_2, 3, 512, [128, 128, 512], "3d_", executor_ctx)

        #conv4_x
        conv4, dict4 = self.convolutional_block(iden3_3, 3, 512, [256, 256, 1024], "4a_", 2, executor_ctx)
        iden4_1, dict4_1 = self.identity_block(conv4, 3, 1024, [256, 256, 1024], "4b_", executor_ctx)
        iden4_2, dict4_2 = self.identity_block(iden4_1, 3, 1024, [256, 256, 1024], "4c_", executor_ctx)
        iden4_3, dict4_3 = self.identity_block(iden4_2, 3, 1024, [256, 256, 1024], "4d_", executor_ctx)
        iden4_4, dict4_4 = self.identity_block(iden4_3, 3, 1024, [256, 256, 1024], "4e_", executor_ctx)
        iden4_5, dict4_5 = self.identity_block(iden4_4, 3, 1024, [256, 256, 1024], "4f_", executor_ctx)


        #conv5_x
        conv5, dict5 = self.convolutional_block(iden4_5, 3, 1024, [512, 512, 2048], "5a_", 2, executor_ctx)
        iden5_1, dict5_1 = self.identity_block(conv5, 3, 2048, [512, 512, 2048], "5b_", executor_ctx)
        iden5_2, dict5_2 = self.identity_block(iden5_1, 3, 2048, [512, 512, 2048], "5c_", executor_ctx)
        pool5 = self.ad.pooling_2d_forward_op(iden5_2, "NCHW", "mean", 0, 0, 1, 1, 7, 7)


        pool5_flat = self.ad.flatten_op(pool5)
        dense6 = self.ad.dense(pool5_flat, W6, b6)
        act6 = self.ad.fullyactivation_forward_op(dense6, "NCHW", "relu")
        drop6 = self.ad.fullydropout_forward_op(act6, "NCHW", self.dropout_rate)

        dense7 = self.ad.dense(drop6, W7, b7)
        y = self.ad.fullyactivation_forward_op(dense7, "NCHW", "softmax")

        loss = self.ad.crossEntropy_loss(y, y_)


        W1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(64, self.image_channel, 7, 7)), executor_ctx)
        W6_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(1*1*2048, 2048)), executor_ctx)
        b6_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(2048)), executor_ctx)
        W7_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(2048, n_class)), executor_ctx)
        b7_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(n_class)), executor_ctx)


        feed_dict = {W1: W1_val, W6: W6_val, W7: W7_val, b6: b6_val, b7: b7_val}
        feed_dict.update(dict2)
        feed_dict.update(dict2_1)
        feed_dict.update(dict2_2)
        feed_dict.update(dict3)
        feed_dict.update(dict3_1)
        feed_dict.update(dict3_2)
        feed_dict.update(dict3_3)
        feed_dict.update(dict4)
        feed_dict.update(dict4_1)
        feed_dict.update(dict4_2)
        feed_dict.update(dict4_3)
        feed_dict.update(dict4_4)
        feed_dict.update(dict4_5)
        feed_dict.update(dict5)
        feed_dict.update(dict5_1)
        feed_dict.update(dict5_2)

        # 只声明，不操作
        executor = self.ad.Executor(loss, y, 0.001, top_control_queue=top_control_queue,
                                    top_message_queue=top_message_queue)

        feed_dict_mv = {}
        for key, value in feed_dict.items():
            m_key = executor.Variable_node_to_mv[key][0]
            m_val = ndarray.array(np.zeros(shape=value.shape), executor_ctx)
            v_key = executor.Variable_node_to_mv[key][1]
            v_val = ndarray.array(np.zeros(shape=value.shape), executor_ctx)
            feed_dict_mv.update({m_key: m_val, v_key: v_val})

        feed_dict.update(feed_dict_mv)
        for i in range(self.num_step):
            print("step", i)

            feed_dict[X] = ndarray.array(X_val, ctx=executor_ctx)
            feed_dict[y_] = ndarray.array(y_val, ctx=executor_ctx)
            res = executor.run(feed_dict=feed_dict)
            loss_val = res[0]
            feed_dict = res[1]


        print("success")


class GPURecord(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(GPU)
        self.f = open("./log/gpu_record.txt", "w+")
        # todo 临时用作释放的计数器
        self.times = 0

    def run(self):
        while True:
            if self.times == 30:
                self.f.close()
                break
            self.times += 1
            time.sleep(1)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            print("time", datetime.datetime.now(),
                  "\tmemory", meminfo.used / 1024 ** 2, file = self.f)  # 已用显存大小

    def stop(self):
        self.f.close()

if __name__ == '__main__':
    # gpu_record = GPURecord()
    global_message_queue = multiprocessing.Queue()
    global_control_queue = multiprocessing.Queue()

    top_control_queue_list = []
    top_message_queue_list = []
    executor_ctx = ndarray.gpu(0)
    print_loss_val_each_epoch = True

    top_control_queue = multiprocessing.Queue()
    top_control_queue_list.append(top_control_queue)
    top_message_queue = multiprocessing.Queue()
    top_message_queue_list.append(top_message_queue)
    job_number = 1

    gpu_num = GPU
    batch_size=4
    num_step = 20

    resNet50 = ResNet50(num_step=num_step, batch_size=batch_size, gpu_num=gpu_num)
    X_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 3, 224, 224))  # number = batch_size  channel = 3  image_size = 224*224
    y_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 1000))  # n_class = 1000


    p1 = Process(target=resNet50.res_net, args=(executor_ctx, top_control_queue, top_message_queue, 1000, X_val, y_val))
    p1.start()
    # p1.join()

    top_control_queue2 = multiprocessing.Queue()
    top_control_queue_list.append(top_control_queue2)
    top_message_queue2 = multiprocessing.Queue()
    top_message_queue_list.append(top_message_queue2)
    job_number += 1

    gpu_num = GPU
    batch_size = 4
    num_step = 20

    resNet50 = ResNet50(num_step=num_step, batch_size=batch_size, gpu_num=gpu_num)
    X_val = np.random.normal(loc=0, scale=0.1,
                             size=(batch_size, 3, 224, 224))  # number = batch_size  channel = 3  image_size = 224*224
    y_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 1000))  # n_class = 1000

    p2 = Process(target=resNet50.res_net, args=(executor_ctx, top_control_queue2, top_message_queue2, 1000, X_val, y_val))
    p2.start()
    # p1.join()

    scheduler = Process(target=mp.multiprocess_init, args=(global_message_queue, global_control_queue))
    scheduler.start()
    # scheduler.join()
    # gpu_record.start()

    while True:
        for i in range(job_number):
            if not top_message_queue_list[i].empty():
                print("job ", i, "message")
                global_message_queue.put([i, top_message_queue_list[i].get()])
        if not global_control_queue.empty():
            global_control = global_control_queue.get()
            for i in range(job_number):
                if i in global_control:
                    print("job ", i, "control")
                    top_control_queue_list[i].put(global_control[i])