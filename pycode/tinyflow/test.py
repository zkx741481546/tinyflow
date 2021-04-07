# """ library to take autodiff and execute a computation graph """
# from __future__ import absolute_import
#
# import threading
# import time
# import numpy as np
# import random
# import queue
# import datetime
# import time
# import numpy as np
# from pycode.tinyflow import ndarray, gpu_op, memoryManager, memoryManagerController
# import random
# import queue
#
# # rand = np.random.RandomState(seed=123)
# # W1_val = np.ones((78400, 4560))
# # ctx_gpu = ndarray.gpu(0)
# # ctx_cpu = ndarray.cpu(0)
# # list_w1 = []
# #
# # w1 = ndarray.array(W1_val, ctx_cpu)
# # w2 = ndarray.empty(w1.shape, ctx_gpu)
# # w1.copyto(w2)
# # print(w2.asnumpy())
# # t1 = time.time()
# #
# # for i in range(100):
# #     w1.copyto(w2)
# #     t2 = time.time()
# #     print(t2 - t1)
# # print("success")
# # for i in range(100):
# #     list_w1.append(ndarray.array(W1_val, ctx_gpu))
# #     del list_w1[i]
#
# # time.sleep(10)
#
# # cudaStream = gpu_op.create_cudaStream()
# # ctx_cpu = ndarray.cpu(0)
# # ctx_gpu = ndarray.gpu(0)
# # w1_np = np.ones((10000, 10000), dtype=np.float32)
# # w1 = ndarray.array(w1_np, ctx_cpu)
# # w2 = ndarray.empty((10000, 10000), ctx_gpu)
# #
# #
# # t1 = datetime.datetime.now()
# # w1.copyto(w2, cudaStream)
# # t2 = datetime.datetime.now()
# # print(w2.asnumpy())
# # print((t2 - t1).microseconds)
#
# executor_ctx = ndarray.gpu(0)
# def test(gpu_ndarray):
#     for i in gpu_ndarray.keys():
#         gpu_ndarray[i] = None
#
# batch_size = 1000
#
#
# rand = np.random.RandomState(seed=123)
# W1_val = rand.normal(scale=0.1, size=(784, 4096))
# W2_val = rand.normal(scale=0.1, size=(4096, 1024))
# # W3_val = rand.normal(scale=0.1, size=(1024, 1024))
# # W4_val = rand.normal(scale=0.1, size=(1024, 1024))
# # W5_val = rand.normal(scale=0.1, size=(1024, 1024))
# # W6_val = rand.normal(scale=0.1, size=(1024, 10))
# #
# # b1_val = rand.normal(scale=0.1, size=(4096))
# # b2_val = rand.normal(scale=0.1, size=(1024))
# # b3_val = rand.normal(scale=0.1, size=(1024))
# # b4_val = rand.normal(scale=0.1, size=(1024))
# # b5_val = rand.normal(scale=0.1, size=(1024))
# # b6_val = rand.normal(scale=0.1, size=(10))
# # W1_val_m = np.zeros(shape=(784, 4096), dtype=np.float32)
# # W2_val_m = np.zeros(shape=(4096, 1024), dtype=np.float32)
# # W3_val_m = np.zeros(shape=(1024, 1024), dtype=np.float32)
# # W4_val_m = np.zeros(shape=(1024, 1024), dtype=np.float32)
# # W5_val_m = np.zeros(shape=(1024, 1024), dtype=np.float32)
# # W6_val_m = np.zeros(shape=(1024, 10), dtype=np.float32)
# # b1_val_m = np.zeros(shape=(4096), dtype=np.float32)
# # b2_val_m = np.zeros(shape=(1024), dtype=np.float32)
# # b3_val_m = np.zeros(shape=(1024), dtype=np.float32)
# # b4_val_m = np.zeros(shape=(1024), dtype=np.float32)
# # b5_val_m = np.zeros(shape=(1024), dtype=np.float32)
# # b6_val_m = np.zeros(shape=(10), dtype=np.float32)
# # W1_val_v = np.zeros(shape=(784, 4096), dtype=np.float32)
# # W2_val_v = np.zeros(shape=(4096, 1024), dtype=np.float32)
# # W3_val_v = np.zeros(shape=(1024, 1024), dtype=np.float32)
# # W4_val_v = np.zeros(shape=(1024, 1024), dtype=np.float32)
# # W5_val_v = np.zeros(shape=(1024, 1024), dtype=np.float32)
# # W6_val_v = np.zeros(shape=(1024, 10), dtype=np.float32)
# # b1_val_v = np.zeros(shape=(4096), dtype=np.float32)
# # b2_val_v = np.zeros(shape=(1024), dtype=np.float32)
# # b3_val_v = np.zeros(shape=(1024), dtype=np.float32)
# # b4_val_v = np.zeros(shape=(1024), dtype=np.float32)
# # b5_val_v = np.zeros(shape=(1024), dtype=np.float32)
# # b6_val_v = np.zeros(shape=(10), dtype=np.float32)
# # X_val = np.zeros(shape=(batch_size, 784), dtype=np.float32)
# # y_val = np.zeros(shape=(batch_size, 10), dtype=np.float32)
# # valid_X_val = np.zeros(shape=(batch_size, 784), dtype=np.float32)
# # valid_y_val = np.zeros(shape=(batch_size, 10), dtype=np.float32)
#
#
# # todo 此处修改回gpu
# W1_val = ndarray.array(W1_val, ctx=executor_ctx)
# W2_val = ndarray.array(W2_val, ctx=executor_ctx)
# # W3_val = ndarray.array(W3_val, ctx=executor_ctx)
# # W4_val = ndarray.array(W4_val, ctx=executor_ctx)
# # W5_val = ndarray.array(W5_val, ctx=executor_ctx)
# # W6_val = ndarray.array(W6_val, ctx=executor_ctx)
# # b1_val = ndarray.array(b1_val, ctx=executor_ctx)
# # b2_val = ndarray.array(b2_val, ctx=executor_ctx)
# # b3_val = ndarray.array(b3_val, ctx=executor_ctx)
# # b4_val = ndarray.array(b4_val, ctx=executor_ctx)
# # b5_val = ndarray.array(b5_val, ctx=executor_ctx)
# # b6_val = ndarray.array(b6_val, ctx=executor_ctx)
# # W1_val_m = ndarray.array(W1_val_m, ctx=executor_ctx)
# # W2_val_m = ndarray.array(W2_val_m, ctx=executor_ctx)
# # W3_val_m = ndarray.array(W3_val_m, ctx=executor_ctx)
# # W4_val_m = ndarray.array(W4_val_m, ctx=executor_ctx)
# # W5_val_m = ndarray.array(W5_val_m, ctx=executor_ctx)
# # W6_val_m = ndarray.array(W6_val_m, ctx=executor_ctx)
# # b1_val_m = ndarray.array(b1_val_m, ctx=executor_ctx)
# # b2_val_m = ndarray.array(b2_val_m, ctx=executor_ctx)
# # b3_val_m = ndarray.array(b3_val_m, ctx=executor_ctx)
# # b4_val_m = ndarray.array(b4_val_m, ctx=executor_ctx)
# # b5_val_m = ndarray.array(b5_val_m, ctx=executor_ctx)
# # b6_val_m = ndarray.array(b6_val_m, ctx=executor_ctx)
# # W1_val_v = ndarray.array(W1_val_v, ctx=executor_ctx)
# # W2_val_v = ndarray.array(W2_val_v, ctx=executor_ctx)
# # W3_val_v = ndarray.array(W3_val_v, ctx=executor_ctx)
# # W4_val_v = ndarray.array(W4_val_v, ctx=executor_ctx)
# # W5_val_v = ndarray.array(W5_val_v, ctx=executor_ctx)
# # W6_val_v = ndarray.array(W6_val_v, ctx=executor_ctx)
# # b1_val_v = ndarray.array(b1_val_v, ctx=executor_ctx)
# # b2_val_v = ndarray.array(b2_val_v, ctx=executor_ctx)
# # b3_val_v = ndarray.array(b3_val_v, ctx=executor_ctx)
# # b4_val_v = ndarray.array(b4_val_v, ctx=executor_ctx)
# # b5_val_v = ndarray.array(b5_val_v, ctx=executor_ctx)
# # b6_val_v = ndarray.array(b6_val_v, ctx=executor_ctx)
# # X_val = ndarray.array(X_val, ctx=executor_ctx)
# # y_val = ndarray.array(y_val, ctx=executor_ctx)
#
# feed_dict = {
#     1: W1_val,
#     2: W2_val
# }
#
# test(feed_dict)
#
# time.sleep(10)
#
# print(W1_val)
#
#
#

import sys

class Dog(object):
    pass

def f(a):
    b = [a[0]]
    b = []
    print(sys.getrefcount(a[0]))

c = Dog()

a = [c]
del c
f(a)



