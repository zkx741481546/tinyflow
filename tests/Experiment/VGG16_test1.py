import six.moves.cPickle as pickle
import numpy as np
import random, imp, time, os, gzip, datetime
from tests.Experiment import record_GPU



tinyflow_path = "../../pycode/tinyflow/"

class VGG16():
    def __init__(self, num_step, type, batch_size, gpu_num, file_name):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
        self.gpu_num = gpu_num

        self.dropout_rate = 0.5
        self.image_channel = 3
        self.image_size = 224
        self.num_step = num_step
        self.batch_size = batch_size
        self.is_capu = False
        self.f1 = open('./log/' + 'type' + str(type) + '_VGG16_' + file_name + '_record_1.txt', 'w')
        self.f3 = open('./log/' + 'type' + str(type) + '_VGG16_' + file_name + '_record_3.txt', 'w')
        self.f6 = open('./log/' + 'type' + str(type) + '_VGG16_' + file_name + '_record_6.txt', 'w')
        self.f7 = open('./log/' + 'type' + str(type) + '_VGG16_' + file_name + '_record_7.txt', 'w')

        self.type = type
        if type == 0:
            self.autodiff_name = "autodiff.py"
            self.TrainExecute_name = "TrainExecuteAdam.py"
        elif type == 1:
            self.autodiff_name = "autodiff_capu.py"
            self.TrainExecute_name = "TrainExecuteAdam_Capu.py"
            self.is_capu=True
        elif type == 2:
            self.autodiff_name = "autodiff_vdnn.py"
            self.TrainExecute_name = "TrainExecuteAdam_vDNNall.py"
        elif type == 3:
            self.autodiff_name = "autodiff.py"
            self.TrainExecute_name = "TrainExecuteAdam.py"
        self.ad = imp.load_source(self.autodiff_name, tinyflow_path + self.autodiff_name)
        self.TrainExecute = imp.load_source(self.autodiff_name, tinyflow_path + self.TrainExecute_name)

    def vgg16(self, num_step ,n_class, X_val, y_val):
        start_time = datetime.datetime.now()

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

        W1_1_val = np.random.normal(0.0, 0.1, (64, self.image_channel, 3, 3))
        W1_2_val = np.random.normal(0.0, 0.1, (64, 64, 3, 3))
        W2_1_val = np.random.normal(0.0, 0.1, (128, 64, 3, 3))
        W2_2_val = np.random.normal(0.0, 0.1, (128, 128, 3, 3))
        W3_1_val = np.random.normal(0.0, 0.1, (256, 128, 3, 3))
        W3_2_val = np.random.normal(0.0, 0.1, (256, 256, 3, 3))
        W3_3_val = np.random.normal(0.0, 0.1, (256, 256, 3, 3))
        W4_1_val = np.random.normal(0.0, 0.1, (512, 256, 3, 3))
        W4_2_val = np.random.normal(0.0, 0.1, (512, 512, 3, 3))
        W4_3_val = np.random.normal(0.0, 0.1, (512, 512, 3, 3))
        W5_1_val = np.random.normal(0.0, 0.1, (512, 512, 3, 3))
        W5_2_val = np.random.normal(0.0, 0.1, (512, 512, 3, 3))
        W5_3_val = np.random.normal(0.0, 0.1, (512, 512, 3, 3))
        W6_val = np.random.normal(0.0, 0.1, (512 * int(self.image_size/32) * int(self.image_size/32), 4096))
        W7_val = np.random.normal(0.0, 0.1, (4096, 4096)) * 0.001
        W8_val = np.random.normal(0.0, 0.1, (4096, n_class)) * 0.001
        b6_val = np.ones(4096) * 0.1
        b7_val = np.ones(4096) * 0.1
        b8_val = np.ones(n_class) * 0.1

        # ctx = ndarray.gpu(0)
        # for i in range(16):
        #     filters_val[i] = ndarray.array(filters_val[i], ctx)
        # for i in range(16):
        #     b_val[i] = ndarray.array(b_val[i], ctx)

        aph = 0.001
        if self.is_capu == True:
            t = self.TrainExecute.TrainExecutor(loss, aph, self.gpu_num)
        else:
            t = self.TrainExecute.TrainExecutor(loss, aph)
        t.init_Variable(
            {W1_1: W1_1_val, W1_2: W1_2_val, W2_1: W2_1_val, W2_2: W2_2_val
                , W3_1: W3_1_val, W3_2: W3_2_val, W3_3: W3_3_val
                , W4_1: W4_1_val, W4_2: W4_2_val, W4_3: W4_3_val
                , W5_1: W5_1_val, W5_2: W5_2_val, W5_3: W5_3_val
                , W6: W6_val, W7: W7_val, W8: W8_val
                , b6: b6_val, b7: b7_val, b8: b8_val})

        for i in range(num_step):
            time1 = datetime.datetime.now()
            print("step", i)
            t.run({X: X_val, y_: y_val})

            time2 = datetime.datetime.now()
            print("epoch", i + 1, "use", time2 - time1
                  , "\tstart", time1, "\tend", time2, file=self.f1)


        start_finish_time = t.get_start_finish_time()
        print("start_time ", start_time, "\nstart_finish_time", start_finish_time, file=self.f3)
        hit_count, swap_count = t.get_hit()
        print("hit_count ", hit_count, "\nswap_count", swap_count, file=self.f6)
        node_order = t.get_node_order()
        for i in node_order:
            print(i, file=self.f7)

        self.f1.close()
        self.f3.close()
        self.f6.close()
        self.f7.close()

        return 0


def run(num_step, type, batch_size, gpu_num, file_name):
    X_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 3, 224, 224))  # number = batch_size  channel = 3  image_size = 224*224
    y_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 1000))  # n_class = 1000

    record = record_GPU.record("VGG16", type, gpu_num, file_name)
    record.start()

    print("VGG16" + " type" + str(type) + " start")

    vgg16 = VGG16(num_step, type, batch_size, gpu_num, file_name)
    vgg16.vgg16(num_step=num_step, n_class=1000, X_val=X_val, y_val=y_val)

    print("VGG16" + " type" + str(type) + " finish")

    record.stop()

# vgg16 = VGG16(num_step=10, type=2, batch_size=4, gpu_num=0, file_name="")

