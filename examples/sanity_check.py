import numpy as np
from quagga.blocks import Ravel
from quagga.blocks import FakeDataBlock
from quagga.blocks import EmbeddingBlock
from quagga.blocks import LogisticRegressionCe


class Network(object):
    def __init__(self):
        embedding_init = lambda: np.random.rand(4, 2).astype(np.float32)
        embedding_init.nrows = 4
        embedding_init.ncols = 2
        log_reg_init = lambda: np.zeros((1, 12)).astype(np.float32)

        data_block = FakeDataBlock(device_id=0)
        embd_block = EmbeddingBlock(embedding_init, data_block.data, device_id=1)
        ravel_block = Ravel(embd_block.output, device_id=0)
        log_reg = LogisticRegressionCe(log_reg_init, ravel_block.output, data_block.y, device_id=1)
        self.blocks = [data_block, embd_block, log_reg]

    def fprop(self):
        for block in self.blocks:
            block.fprop()


model = Network()
model.fprop()
model.fprop()
model.fprop()