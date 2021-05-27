import numpy as np
from pycode.tinyflow import mainV2 as mp
from pycode.tinyflow import autodiff as ad
from pycode.tinyflow import ndarray
import threading, pynvml, multiprocessing, os, datetime, time
from multiprocessing import Process
from util import *


with open('./log_path.txt', 'r') as f:
    log_path = f.readlines()[0]
    if not os.path.exists(log_path):
        os.makedirs(log_path)
GPU = load_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = f'{GPU}'
class DenseNet121():
    def __init__(self, num_step, type, batch_size, gpu_num):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
        self.gpu_num = gpu_num

        self.n_filter = 32 # growth rate
        self.image_channel = 3
        self.dropout_rate = 0.2
        self.num_step = num_step
        self.batch_size = batch_size

        self.type = type
        self.ad = ad


    def bottleneck_layer(self, inputs, in_filter, layer_name, executor_ctx):

        W1 = self.ad.Variable(layer_name + "_W1")
        W2 = self.ad.Variable(layer_name + "_W2")


        # 1*1 conv
        bn1 = self.ad.bn_forward_op(inputs, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")
        conv1 = self.ad.convolution_2d_forward_op(act1, W1, "NCHW", "SAME", 1, 1)
        drop1 = self.ad.dropout_forward_op(conv1, "NCHW", self.dropout_rate)


        # 3*3 conv
        bn2 = self.ad.bn_forward_op(drop1, "NCHW", "pre_activation")
        act2 = self.ad.activation_forward_op(bn2, "NCHW", "relu")
        conv2 = self.ad.convolution_2d_forward_op(act2, W2, "NCHW", "SAME", 1, 1)
        drop2 = self.ad.dropout_forward_op(conv2, "NCHW", self.dropout_rate)


        W1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(4 * self.n_filter, in_filter, 1, 1)), executor_ctx)  # kernel_size=1*1
        W2_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(self.n_filter, 4 * self.n_filter, 3, 3)), executor_ctx)  # kernel_size=3*3

        dict = {W1: W1_val, W2: W2_val}
        return drop2, dict, self.n_filter

    def dense_block(self, inputs, in_filter, nb_layers, block_name, executor_ctx):

        x1 = inputs
        dict1 = {}
        for i in range(nb_layers):
            x2, dict2, out_filter = self.bottleneck_layer(x1, in_filter, block_name + "_bottleneck" + str(i), executor_ctx)
            x1 = self.ad.concat_forward_op(x1, x2)
            in_filter = in_filter + out_filter
            dict1.update(dict2)
        return x1, dict1, in_filter

    def transition_layer(self, inputs, in_filter, layer_name, executor_ctx):

        W1 = self.ad.Variable(layer_name + "_W1")
        bn1 = self.ad.bn_forward_op(inputs, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")
        conv1 = self.ad.convolution_2d_forward_op(act1, W1, "NCHW", "SAME", 1, 1)
        drop1 = self.ad.dropout_forward_op(conv1, "NCHW", self.dropout_rate)
        pool0 = self.ad.pooling_2d_forward_op(drop1, "NCHW", "mean", 0, 0, 2, 2, 2, 2)  # stride=2   pool_size=2*2

        W1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(int(0.5*in_filter), in_filter, 1, 1)) , executor_ctx) # kernel_size=1*1
        dict = {W1: W1_val}
        return pool0, dict, int(0.5*in_filter)



    def dense_net(self, executor_ctx, top_control_queue, top_message_queue ,n_class, X_val, y_val):
        gpu_record = GPURecord(log_path)
        X = self.ad.Placeholder("X")
        y_ = self.ad.Placeholder("y_")
        W0 = self.ad.Variable("W0")
        W1 = self.ad.Variable("W1")
        b1 = self.ad.Variable("b1")

        conv0 = self.ad.convolution_2d_forward_op(X, W0, "NCHW", "SAME", 2, 2) # stride=2
        pool0 = self.ad.pooling_2d_forward_op(conv0, "NCHW", "max", 1, 1, 2, 2, 3, 3) # stride=2   pool_size=3*3

        dense_1, dict_1, out_filter1 = self.dense_block(inputs=pool0, in_filter=2 * self.n_filter, nb_layers=6, block_name="dense1", executor_ctx=executor_ctx)
        transition_1, dict_2, out_filter2 = self.transition_layer(inputs=dense_1, in_filter=out_filter1, layer_name="trans1", executor_ctx=executor_ctx)

        dense_2, dict_3, out_filter3 = self.dense_block(inputs=transition_1, in_filter=out_filter2, nb_layers=12, block_name="dense2", executor_ctx=executor_ctx)
        transition_2, dict_4, out_filter4 = self.transition_layer(inputs=dense_2, in_filter=out_filter3, layer_name="trans2", executor_ctx=executor_ctx)

        dense_3, dict_5, out_filter5 = self.dense_block(inputs=transition_2, in_filter=out_filter4, nb_layers=24, block_name="dense3", executor_ctx=executor_ctx)
        transition_3, dict_6, out_filter6 = self.transition_layer(inputs=dense_3, in_filter=out_filter5, layer_name="trans3", executor_ctx=executor_ctx)

        dense_4, dict_7, out_filter7 = self.dense_block(inputs=transition_3, in_filter=out_filter6, nb_layers=16, block_name="dense4", executor_ctx=executor_ctx)

        bn1 = self.ad.bn_forward_op(dense_4, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")
        pool1 = self.ad.pooling_2d_forward_op(act1, "NCHW", "mean", 0, 0, 1, 1, 7, 7) # global_pool

        flat = self.ad.flatten_op(pool1)
        dense = self.ad.dense(flat, W1, b1)
        y = self.ad.fullyactivation_forward_op(dense, "NCHW", "softmax")

        loss = self.ad.crossEntropy_loss(y, y_)



        W0_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(2 * self.n_filter, self.image_channel, 7, 7)), executor_ctx)  # n_filter   n_channel=3   kernel_size=7*7
        W1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(out_filter7, n_class)), executor_ctx)
        b1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(n_class)), executor_ctx)


        executor = self.ad.Executor(loss, y, 0.001, top_control_queue=top_control_queue,
                                    top_message_queue=top_message_queue)

        feed_dict = {W0: W0_val, W1: W1_val, b1: b1_val}
        feed_dict.update(dict_1)
        feed_dict.update(dict_2)
        feed_dict.update(dict_3)
        feed_dict.update(dict_4)
        feed_dict.update(dict_5)
        feed_dict.update(dict_6)
        feed_dict.update(dict_7)

        feed_dict_mv = {}
        for key, value in feed_dict.items():
            m_key = executor.Variable_node_to_mv[key][0]
            m_val = ndarray.array(np.zeros(shape=value.shape), executor_ctx)
            v_key = executor.Variable_node_to_mv[key][1]
            v_val = ndarray.array(np.zeros(shape=value.shape), executor_ctx)
            feed_dict_mv.update({m_key: m_val, v_key: v_val})

        feed_dict.update(feed_dict_mv)
        f1 = open(f"{log_path}/gpu_time.txt", "w+")

        for i in range(self.num_step):
            print("step", i)
            if i==5:
                gpu_record.start()
                start_time = time.time()
            if i==10:
                gpu_record.stop()
                f1.write(f'time_cost:{time.time() - start_time}')
                f1.flush()
                f1.close()
            feed_dict[X] = ndarray.array(X_val, ctx=executor_ctx)
            feed_dict[y_] = ndarray.array(y_val, ctx=executor_ctx)
            res = executor.run(feed_dict=feed_dict)
            loss_val = res[0]
            feed_dict = res[1]

        print("success")
        return 0


if __name__ == '__main__':
    gpu_record = GPURecord()
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
    denseNet = DenseNet121(num_step=num_step, type=3, batch_size=batch_size, gpu_num=gpu_num) #type=3代表学长的方法
    X_val = np.random.normal(loc=0, scale=0.1,
                             size=(batch_size, 3, 224, 224))  # number = batch_size  channel = 3  image_size = 224*224
    y_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 1000))  # n_class = 1000

    p1 = Process(target=denseNet.dense_net, args=(executor_ctx, top_control_queue, top_message_queue, 1000, X_val, y_val))
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
    denseNet = DenseNet121(num_step=num_step, type=3, batch_size=batch_size, gpu_num=gpu_num)  # type=3代表学长的方法
    X_val = np.random.normal(loc=0, scale=0.1,
                             size=(batch_size, 3, 224, 224))  # number = batch_size  channel = 3  image_size = 224*224
    y_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 1000))  # n_class = 1000

    p2 = Process(target=denseNet.dense_net,
                 args=(executor_ctx, top_control_queue2, top_message_queue2, 1000, X_val, y_val))
    p2.start()

    if 'schedule' in log_path:
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



