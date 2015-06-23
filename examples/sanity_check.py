import numpy as np
from quagga.blocks import FakeDataBlock
from quagga.blocks import EmbeddingBlock


class Network(object):
    def __init__(self):
        embedding_init = lambda: np.random.rand(2, 3).astype(np.float32)
        embedding_init.nrows = 2
        embedding_init.ncols = 3
        data_block = FakeDataBlock(device_id=0)
        embd_block = EmbeddingBlock(embedding_init, data_block.data, 3, device_id=1)
        self.blocks = [data_block, embd_block]

    def fprop(self):
        for block in self.blocks:
            block.fprop()


model = Network()
model.fprop()
model.fprop()
model.fprop()