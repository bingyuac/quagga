import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import Connector


class FakeDataBlock(object):
    def __init__(self, device_id):
        self.device_id = device_id
        self.rng = np.random.RandomState(42)
        self.context = Context(device_id)
        self.data = Connector(Matrix.empty(1, 3, 'int', device_id), self.context)
        self.y = Connector(Matrix.empty(1, 1, 'float', device_id), self.context)

    def fprop(self):
        data = self.rng.randint(2, size=(1, 3)).astype(np.int32)
        self.data.to_device(self.context, data)
        if np.sum(data) == 2:
            y = np.array([[1]], dtype=np.float32)
        else:
            y = np.array([[0]], dtype=np.float32)
        self.y.to_device(self.context, y)
        self.data.block_users()
        self.y.block_users()

    def bprop(self):
        pass