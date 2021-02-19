
# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(3)
# print(pynvml.nvmlDeviceGetPcieThroughput(handle, 0))
# print(pynvml.nvmlDeviceGetPcieThroughput(handle, 1))
# print(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
# print(pynvml.nvmlDeviceGetPerformanceState(handle))
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', type=int, default=None)
# args = parser.parse_args()
# gpu = args.gpu
def run(gpu):
    import os
    import time
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import numpy as np
    from pycode.tinyflow import gpu_op
    from pycode.tinyflow import ndarray
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    W1_val = np.ones((10000, 100))

    X_val = np.ones((10000, 10000))
    ctx = ndarray.gpu(0)

    z = ndarray.array(np.ones((100000, 20000)), ctx)


    W1_val = ndarray.array(W1_val, ctx=ctx)

    X_val = ndarray.array(X_val, ctx=ctx)

    out_val = ndarray.empty((10000, 100), ctx=ctx)

    i = 0
    while True:
        i += 1
        gpu_op.matrix_multiply(X_val, False, W1_val, False, out_val)
        print(f'GPU:{gpu}, iter:{i}')
        time.sleep(0.005)




run(7)
