import threading
import datetime
import time

import pynvml
def load_gpu():
    with open('../../GPU.txt') as f:
        c = f.readlines()
        c = c[0].strip()
        GPU = int(c)
    return GPU

class GPURecord(threading.Thread):
    def __init__(self, log_path):
        threading.Thread.__init__(self)
        pynvml.nvmlInit()
        GPU = load_gpu()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(GPU)
        self.f = open(f"{log_path}/gpu_record.txt", "w+")
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
            self.memory_used = meminfo.used / 1024 ** 2
            if self.memory_used>self.max_gpu_memory:
                self.max_gpu_memory = self.memory_used


    def stop(self):
        self.flag = False
        time.sleep(0.01)
        print("time", datetime.datetime.now(),
              "\tmemory", self.memory_used,
              "\tmax_memory_used", self.max_gpu_memory,
              "\tretained_memory_used", self.memory_used - self.base_used,
              "\tretained_max_memory_used", self.max_gpu_memory - self.base_used, file=self.f)  # 已用显存大小
        self.f.flush()
        self.f.close()