import pynvml
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(3)
# print(pynvml.nvmlDeviceGetPcieThroughput(handle, 0))
# print(pynvml.nvmlDeviceGetPcieThroughput(handle, 1))
while True:
    print(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
# print(pynvml.nvmlDeviceGetPerformanceState(handle))
