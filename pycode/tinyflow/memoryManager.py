import threading
import queue
from . import ndarray, gpu_op



class MemoryManager(threading.Thread):
    def __init__(self, will_do_queue:queue.Queue, have_done_queue:queue.Queue):
        threading.Thread.__init__(self)
        self.will_do_queue = will_do_queue
        self.have_done_queue = have_done_queue
        # todo hard code with device id again, may need to change
        self.cpu_ctx = ndarray.cpu(0)
        self.gpu_ctx = ndarray.gpu(0)
        self.cudaSwapStream = gpu_op.create_cudaStream()
        self.flag = 0

    def run(self):
        while (True):
            try:
                node = self.will_do_queue.get(block=True, timeout=10)
            except:
                break

            node_index = node[0]
            node_ndarray = node[1]
            ''':type:ndarray.NDArray'''
            node_ndarray_new = None
            if ndarray.is_gpu_ctx(node_ndarray.ctx):
                node_ndarray_new = ndarray.empty(node_ndarray.shape, self.cpu_ctx)
            else:
                node_ndarray_new = ndarray.empty(node_ndarray.shape, self.gpu_ctx)

            if (isinstance(node_ndarray_new, int)):
                print("内存超限")
                assert 0

            node_ndarray.copyto(node_ndarray_new, self.cudaSwapStream)
            self.have_done_queue.put((node_index, node_ndarray_new))


