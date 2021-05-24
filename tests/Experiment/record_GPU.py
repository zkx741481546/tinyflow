import threading, pynvml, datetime, time

class record(threading.Thread):
    def __init__(self, net, type, gpu_num, file_name):
        threading.Thread.__init__(self)
        self.gpu_num = gpu_num
        # self.handle = handle
        self.f2 = open('./log/' + 'type' + str(type) + '_' + net + '_'+ file_name + '_record_2.txt', 'w')
        self.flag = 0
    def run(self):
        while(True):
            if self.flag == 1:
                break
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_num)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # print(meminfo.total / 1024 ** 2)  # 总的显存大小
            print("time", datetime.datetime.now(),
                  "\tmemory", meminfo.used / 1024 ** 2, file=self.f2)  # 已用显存大小
            # print(meminfo.free / 1024 ** 2)  # 剩余显存大小

            # time.sleep(0.1)
    def stop(self):
        self.flag = 1
        time.sleep(1)
        self.f2.close()