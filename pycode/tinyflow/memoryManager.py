import threading
import queue
from . import ndarray, gpu_op



class MemoryManager(threading.Thread):
    def __init__(self, will_do_queue:queue.Queue, have_done_queue:queue.Queue, index_to_cpu_map, index_to_gpu_map):
        threading.Thread.__init__(self)
        self.will_do_queue = will_do_queue
        self.have_done_queue = have_done_queue
        # todo hard code with device id again, may need to change
        self.cpu_ctx = ndarray.cpu(0)
        self.gpu_ctx = ndarray.gpu(0)
        self.index_to_cpu_map = index_to_cpu_map
        self.index_to_gpu_map = index_to_gpu_map
        self.cudaSwapStream = gpu_op.create_cudaStream()


    def run(self):
        while (True):
            node = self.will_do_queue.get(block=True)
            node_index = node[0]
            move_to_gpu = node[1]
            node_ndarray_new = None

            if move_to_gpu == 0:
                node_ndarray = self.index_to_gpu_map[node_index]
                node_ndarray_new = ndarray.empty(node_ndarray.shape, self.cpu_ctx)
                node_ndarray.copyto(node_ndarray_new, self.cudaSwapStream)
                self.index_to_cpu_map[node_index] = node_ndarray_new
            else:
                node_ndarray = self.index_to_cpu_map[node_index]
                node_ndarray_new = ndarray.empty(node_ndarray.shape, self.gpu_ctx)
                node_ndarray.copyto(node_ndarray_new, self.cudaSwapStream)
                self.index_to_gpu_map[node_index] = node_ndarray_new






