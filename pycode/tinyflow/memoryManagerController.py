import threading
import queue
import time
from . import ndarray, gpu_op, memoryManager


class MemoryManagerController(threading.Thread):
    def __init__(self, control_queue: queue.Queue, have_done_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.will_do_queue = queue.Queue()
        self.have_done_queue = have_done_queue
        self.control_queue = control_queue
        # todo hard code with device id again, may need to change
        self.cpu_ctx = ndarray.cpu(0)
        self.gpu_ctx = ndarray.gpu(0)
        self.memoryManager = memoryManager.MemoryManager(self.will_do_queue, self.have_done_queue)
        self.memoryManager.start()

    def run(self):
        while True:
            # todo 接口内容：wait_time: 距离上一次swap的间隔时间，node_index和node_ndarray同Manager中的定义
            # todo 在此处检查当前移动是否需要，即检查是否已经在对应的ctx中，加入变量move_to_gpu
            # (wait_time, node_index, node_ndarray, move_to_gpu)
            control_message = self.control_queue.get(block=True)
            wait_time = control_message[0]
            node_index = control_message[1]
            node_ndarray = control_message[2]
            move_to_gpu = control_message[3]
            if move_to_gpu and ndarray.is_gpu_ctx(node_ndarray.ctx):
                continue
            if not move_to_gpu and not ndarray.is_gpu_ctx(node_ndarray.ctx):
                continue
            time.sleep(wait_time)
            self.will_do_queue.put((node_index, node_ndarray))






