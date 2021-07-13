import os
import sys

import numpy as np
from pycode.tinyflow import ndarray

sys.path.append('../../')
from tests.Experiment import VGG16_test, ResNet50_test, DenseNet_test, InceptionV3_test, InceptionV4_test
import random, time

from tests.Experiment.log.result import get_result, get_vanilla_max_memory
import pickle as pkl
from line_profiler import LineProfiler
from pycode.tinyflow.TrainExecuteAdam_vDNNconv import TrainExecutor as vdnnExecutor
from pycode.tinyflow.TrainExecuteAdam_Capu import TrainExecutor as CapuchinExecutor
from pycode.tinyflow.TrainExecuteAdam import TrainExecutor as VanillaTrainExecutor

gpu = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

t = 2
def main():
    vgg16 = VGG16_test.VGG16(num_step=10, type=t, batch_size=16, gpu_num=gpu, path='test', file_name='test', n_class=1000, need_tosave=0)
    vgg16.run()


if __name__ == '__main__':
    profiler = LineProfiler()
    if t == 0:
        profiler.add_function(VanillaTrainExecutor.run)
    elif t==1:
        profiler.add_function(CapuchinExecutor.run)
        profiler.add_function(CapuchinExecutor.clear)
    else:
        profiler.add_function(vdnnExecutor.run)

    profiler_wrapper = profiler(main)
    res = profiler_wrapper()
    profiler.print_stats()
    # main()
