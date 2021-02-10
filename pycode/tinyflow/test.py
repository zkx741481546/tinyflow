from __future__ import absolute_import
import time
import numpy as np
from pycode.tinyflow import ndarray, gpu_op, memoryManager, memoryManagerController
import random
import queue

rand = np.random.RandomState(seed=123)
W1_val = rand.normal(scale=0.1, size=(7840, 2560))
ctx_gpu = ndarray.gpu(0)
ctx_cpu = ndarray.cpu(0)
list_w1 = []
# for i in range(100):
#     w1 = ndarray.array(W1_val, ctx_gpu)
#     w1 = ndarray.array(W1_val, ctx_cpu)

# for i in range(100):
#     list_w1.append(ndarray.array(W1_val, ctx_gpu))
#     del list_w1[i]

# time.sleep(10)

