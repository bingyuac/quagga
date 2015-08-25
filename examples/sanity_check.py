import numpy as np
from quagga import Model
from quagga.blocks import Ravel
from quagga.blocks import DotBlock
from quagga.blocks import FakeDataBlock
from quagga.blocks import EmbeddingBlock
from quagga.blocks import LogisticRegressionCe
from quagga.optimizers import SgdOptimizer
from quagga.optimizers.policy import FixedLearningRatePolicy
from quagga.optimizers.interruption import ValidationInterruption


if __name__ == '__main__':
    embedding_init = lambda: np.random.rand(4, 2).astype(np.float32)
    dense_init = lambda: (0.05 * np.random.rand(20, 12)).astype(np.float32)
    log_reg_init = lambda: (0.05 * np.random.rand(1, 20)).astype(np.float32)

    data_block = FakeDataBlock(device_id=0)
    embd_block = EmbeddingBlock(embedding_init, data_block.data, device_id=1)
    ravel_block = Ravel(embd_block.output, device_id=0)
    dense_block = DotBlock(dense_init, ravel_block.output, 'tanh', device_id=1)
    log_reg = LogisticRegressionCe(log_reg_init, dense_block.output, data_block.y, device_id=0)

    model = Model(data_block, embd_block, ravel_block, dense_block, log_reg)
    learning_rate_policy = FixedLearningRatePolicy(0.1)
    sgd = SgdOptimizer(20000, learning_rate_policy, model)
    sgd.add_interruption(ValidationInterruption(1000, 1000, 'test.log', model))
    sgd.optimize()