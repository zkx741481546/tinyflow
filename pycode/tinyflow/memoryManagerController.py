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
            control_message = self.control_queue.get(block=True)
            wait_time = control_message[0]
            node_index = control_message[1]
            node_ndarray = control_message[2]
            time.sleep(wait_time)
            self.will_do_queue.put((node_index, node_ndarray))






