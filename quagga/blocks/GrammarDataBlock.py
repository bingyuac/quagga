import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class GrammarDataBlock(object):
    def __init__(self, device_id):
        self.device_id = device_id
        self.rng = np.random.RandomState(42)
        self.context = Context(device_id)
        self.max_buffer_size = 2000
        self.data = Connector(Matrix.empty(1, self.max_buffer_size, 'int', device_id), self.context)
        self.y = Connector(Matrix.empty(1, 1, 'float', device_id), self.context)
        self.i = -1

    def fprop(self):
        self.i = (self.i + 1) % 2
        n = self.rng.randint(1, self.max_buffer_size / 2)
        if self.i == 0:
            data = np.array(n * [0] + n * [1], order='F', dtype=np.int32)
            y = np.array([[1.0]], dtype=np.float32)
        else:
            data = np.array((n+1) * [0] + (n-1) * [1], order='F', dtype=np.int32)
            y = np.array([[0.0]], dtype=np.float32)
        self.data.to_device(self.context, data)
        self.y.to_device(self.context, y)
        self.data.fprop()
        self.y.fprop()