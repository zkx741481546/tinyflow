import numpy as np
from pycode.tinyflow import mainV2 as mp
from pycode.tinyflow import autodiff as ad
from pycode.tinyflow.tools import *
from pycode.tinyflow import ndarray
import threading, pynvml, multiprocessing, os, datetime, time
from multiprocessing import Process

GPU = load_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = f'{GPU}'
class VGG16():
    def __init__(self, num_step, batch_size, gpu_num):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
        self.gpu_num = gpu_num

        self.dropout_rate = 0.5
        self.image_channel = 3
        self.image_size = 224
        self.num_step = num_step
        self.batch_size = batch_size

        self.ad = ad

    def vgg16(self, executor_ctx, top_control_queue, top_message_queue, n_class, X_val, y_val):
        gpu_record = GPURecord()
        X = self.ad.Placeholder("X")
        y_ = self.ad.Placeholder("y_")
        W1_1 = self.ad.Variable("W1_1")
        W1_2 = self.ad.Variable("W1_2")
        W2_1 = self.ad.Variable("W2_1")
        W2_2 = self.ad.Variable("W2_2")
        W3_1 = self.ad.Variable("W3_1")
        W3_2 = self.ad.Variable("W3_2")
        W3_3 = self.ad.Variable("W3_3")
        W4_1 = self.ad.Variable("W4_1")
        W4_2 = self.ad.Variable("W4_2")
        W4_3 = self.ad.Variable("W4_3")
        W5_1 = self.ad.Variable("W5_1")
        W5_2 = self.ad.Variable("W5_2")
        W5_3 = self.ad.Variable("W5_3")
        W6 = self.ad.Variable("W6")
        W7 = self.ad.Variable("W7")
        W8 = self.ad.Variable("W8")
        b6 = self.ad.Variable("b6")
        b7 = self.ad.Variable("b7")
        b8 = self.ad.Variable("b8")

        # conv 1
        conv1_1 = self.ad.convolution_2d_forward_op(X, W1_1, "NCHW", "SAME", 1, 1)
        act1_1 = self.ad.activation_forward_op(conv1_1, "NCHW", "relu")

        conv1_2 = self.ad.convolution_2d_forward_op(act1_1, W1_2, "NCHW", "SAME", 1, 1)
        act1_2 = self.ad.activation_forward_op(conv1_2, "NCHW", "relu")
        pool1 = self.ad.pooling_2d_forward_op(act1_2, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # conv 2
        conv2_1 = self.ad.convolution_2d_forward_op(pool1, W2_1, "NCHW", "SAME", 1, 1)
        act2_1 = self.ad.activation_forward_op(conv2_1, "NCHW", "relu")
        conv2_2 = self.ad.convolution_2d_forward_op(act2_1, W2_2, "NCHW", "SAME", 1, 1)
        act2_2 = self.ad.activation_forward_op(conv2_2, "NCHW", "relu")
        pool2 = self.ad.pooling_2d_forward_op(act2_2, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # conv 3
        conv3_1 = self.ad.convolution_2d_forward_op(pool2, W3_1, "NCHW", "SAME", 1, 1)
        act3_1 = self.ad.activation_forward_op(conv3_1, "NCHW", "relu")
        conv3_2 = self.ad.convolution_2d_forward_op(act3_1, W3_2, "NCHW", "SAME", 1, 1)
        act3_2 = self.ad.activation_forward_op(conv3_2, "NCHW", "relu")
        conv3_3 = self.ad.convolution_2d_forward_op(act3_2, W3_3, "NCHW", "SAME", 1, 1)
        act3_3 = self.ad.activation_forward_op(conv3_3, "NCHW", "relu")
        pool3 = self.ad.pooling_2d_forward_op(act3_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # conv 4
        conv4_1 = self.ad.convolution_2d_forward_op(pool3, W4_1, "NCHW", "SAME", 1, 1)
        act4_1 = self.ad.activation_forward_op(conv4_1, "NCHW", "relu")
        conv4_2 = self.ad.convolution_2d_forward_op(act4_1, W4_2, "NCHW", "SAME", 1, 1)
        act4_2 = self.ad.activation_forward_op(conv4_2, "NCHW", "relu")
        conv4_3 = self.ad.convolution_2d_forward_op(act4_2, W4_3, "NCHW", "SAME", 1, 1)
        act4_3 = self.ad.activation_forward_op(conv4_3, "NCHW", "relu")
        pool4 = self.ad.pooling_2d_forward_op(act4_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # conv 5
        conv5_1 = self.ad.convolution_2d_forward_op(pool4, W5_1, "NCHW", "SAME", 1, 1)
        act5_1 = self.ad.activation_forward_op(conv5_1, "NCHW", "relu")
        conv5_2 = self.ad.convolution_2d_forward_op(act5_1, W5_2, "NCHW", "SAME", 1, 1)
        act5_2 = self.ad.activation_forward_op(conv5_2, "NCHW", "relu")
        conv5_3 = self.ad.convolution_2d_forward_op(act5_2, W5_3, "NCHW", "SAME", 1, 1)
        act5_3 = self.ad.activation_forward_op(conv5_3, "NCHW", "relu")
        pool5 = self.ad.pooling_2d_forward_op(act5_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # fc6
        pool5_flat = self.ad.flatten_op(pool5)
        fc6 = self.ad.dense(pool5_flat, W6, b6)
        act6 = self.ad.fullyactivation_forward_op(fc6, "NCHW", "relu")
        drop6 = self.ad.fullydropout_forward_op(act6, "NCHW", self.dropout_rate)

        # fc7
        fc7 = self.ad.dense(drop6, W7, b7)
        act7 = self.ad.fullyactivation_forward_op(fc7, "NCHW", "relu")
        drop7 = self.ad.fullydropout_forward_op(act7, "NCHW", self.dropout_rate)

        # fc8
        fc8 = self.ad.dense(drop7, W8, b8)
        bn8 = self.ad.fullybn_forward_op(fc8, "NCHW")
        y = self.ad.fullyactivation_forward_op(bn8, "NCHW", "softmax")

        loss = self.ad.crossEntropy_loss(y, y_)
        W1_1_val = ndarray.array(np.random.normal(0.0, 0.1, (64, self.image_channel, 3, 3)), executor_ctx)
        W1_2_val = ndarray.array(np.random.normal(0.0, 0.1, (64, 64, 3, 3)), executor_ctx)
        W2_1_val = ndarray.array(np.random.normal(0.0, 0.1, (128, 64, 3, 3)), executor_ctx)
        W2_2_val = ndarray.array(np.random.normal(0.0, 0.1, (128, 128, 3, 3)), executor_ctx)
        W3_1_val = ndarray.array(np.random.normal(0.0, 0.1, (256, 128, 3, 3)), executor_ctx)
        W3_2_val = ndarray.array(np.random.normal(0.0, 0.1, (256, 256, 3, 3)), executor_ctx)
        W3_3_val = ndarray.array(np.random.normal(0.0, 0.1, (256, 256, 3, 3)), executor_ctx)
        W4_1_val = ndarray.array(np.random.normal(0.0, 0.1, (512, 256, 3, 3)), executor_ctx)
        W4_2_val = ndarray.array(np.random.normal(0.0, 0.1, (512, 512, 3, 3)), executor_ctx)
        W4_3_val = ndarray.array(np.random.normal(0.0, 0.1, (512, 512, 3, 3)), executor_ctx)
        W5_1_val = ndarray.array(np.random.normal(0.0, 0.1, (512, 512, 3, 3)), executor_ctx)
        W5_2_val = ndarray.array(np.random.normal(0.0, 0.1, (512, 512, 3, 3)), executor_ctx)
        W5_3_val = ndarray.array(np.random.normal(0.0, 0.1, (512, 512, 3, 3)), executor_ctx)
        W6_val = ndarray.array(np.random.normal(0.0, 0.1, (512 * int(self.image_size/32) * int(self.image_size/32), 4096)), executor_ctx)
        W7_val = ndarray.array(np.random.normal(0.0, 0.1, (4096, 4096)) * 0.001, executor_ctx)
        W8_val = ndarray.array(np.random.normal(0.0, 0.1, (4096, n_class)) * 0.001, executor_ctx)
        b6_val = ndarray.array(np.ones(4096) * 0.1, executor_ctx)
        b7_val = ndarray.array(np.ones(4096) * 0.1, executor_ctx)
        b8_val = ndarray.array(np.ones(n_class) * 0.1, executor_ctx)


        # 只声明，不操作
        executor = self.ad.Executor(loss, y, 0.001, top_control_queue=top_control_queue, top_message_queue=top_message_queue)
        feed_dict = {
            W1_1: W1_1_val,
            W1_2: W1_2_val,
            W2_1: W2_1_val,
            W2_2: W2_2_val,
            W3_1: W3_1_val,
            W3_2: W3_2_val,
            W3_3: W3_3_val,
            W4_1: W4_1_val,
            W4_2: W4_2_val,
            W4_3: W4_3_val,
            W5_1: W5_1_val,
            W5_2: W5_2_val,
            W5_3: W5_3_val,
            W6: W6_val,
            W7: W7_val,
            W8: W8_val,
            b6: b6_val,
            b7: b7_val,
            b8: b8_val
        }
        feed_dict_mv = {}
        for key, value in feed_dict.items():
            m_key = executor.Variable_node_to_mv[key][0]
            m_val = ndarray.array(np.zeros(shape=value.shape), executor_ctx)
            v_key = executor.Variable_node_to_mv[key][1]
            v_val = ndarray.array(np.zeros(shape=value.shape), executor_ctx)
            feed_dict_mv.update({m_key: m_val, v_key: v_val})

        feed_dict.update(feed_dict_mv)
        f1 = open("./log/gpu_time.txt", "w+")
        for i in range(self.num_step):
            print("step", i)
            if i==5:
                gpu_record.start()
                start_time = time.time()
            feed_dict[X] = ndarray.array(X_val, ctx=executor_ctx)
            feed_dict[y_] = ndarray.array(y_val, ctx=executor_ctx)
            res = executor.run(feed_dict=feed_dict)
            loss_val = res[0]
            feed_dict = res[1]
            print(loss_val)
        gpu_record.stop()
        f1.write(f'time_cost:{time.time() - start_time}')
        f1.flush()
        f1.close()

        print("success")
        return 0

class GPURecord(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(GPU)
        self.f = open("./log/gpu_record.txt", "w+")
        # todo 临时用作释放的计数器
        self.times = 0
        self.max_gpu_memory = 0
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.base_used = meminfo.used / 1024 ** 2
        self.flag = True
    def run(self):
        while self.flag:
            # if self.times == 30000:
            #     self.f.close()
            #     break
            self.times += 1
            # time.sleep(0.1)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            memory_used = meminfo.used / 1024 ** 2
            if memory_used>self.max_gpu_memory:
                self.max_gpu_memory = memory_used
            print("time", datetime.datetime.now(),
                  "\tmemory", memory_used,
                  "\tmax_memory_used", self.max_gpu_memory,
                  "\tretained_memory_used", memory_used-self.base_used,
                  "\tretained_max_memory_used", self.max_gpu_memory-self.base_used, file=self.f)  # 已用显存大小
            # self.f.flush()

    def stop(self):
        self.flag = False
        time.sleep(0.001)
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
    batch_size = 2
    num_step = 20
    vgg16 = VGG16(num_step=num_step, batch_size=batch_size, gpu_num=gpu_num)
    X_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 3, 224, 224))  # number = batch_size  channel = 3  image_size = 224*224
    y_val = np.random.randint(low=0, high=1, size=(batch_size, 1000))  # n_class = 1000


    p1 = Process(target=vgg16.vgg16, args=(executor_ctx, top_control_queue, top_message_queue, 1000, X_val, y_val))
    p1.start()

    top_control_queue2 = multiprocessing.Queue()
    top_control_queue_list.append(top_control_queue2)
    top_message_queue2 = multiprocessing.Queue()
    top_message_queue_list.append(top_message_queue2)
    job_number += 1

    gpu_num = GPU
    batch_size = 2
    num_step = 20
    vgg16 = VGG16(num_step=num_step, batch_size=batch_size, gpu_num=gpu_num)
    X_val = np.random.normal(loc=0, scale=0.1,
                             size=(batch_size, 3, 224, 224))  # number = batch_size  channel = 3  image_size = 224*224
    y_val = np.random.randint(low=0, high=1, size=(batch_size, 1000))  # n_class = 1000

    p2 = Process(target=vgg16.vgg16, args=(executor_ctx, top_control_queue2, top_message_queue2, 1000, X_val, y_val))
    p2.start()

    scheduler = Process(target=mp.multiprocess_init, args=(global_message_queue, global_control_queue))
    scheduler.start()


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