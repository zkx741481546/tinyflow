import os
import time

import numpy as np
import six.moves.cPickle as pickle



from pycode.tinyflow import ndarray

os.environ["CUDA_VISIBLE_DEVICES"] = "7"





ctx = ndarray.gpu(0)

z = ndarray.array(np.ones((100000, 20000)), ctx)




while True:
    time.sleep(10000000)