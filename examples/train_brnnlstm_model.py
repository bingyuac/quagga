import numpy as np
from quagga import Model
from quagga.optimizers import SgdOptimizer
from quagga.optimizers.policy import FixedLearningRatePolicy
from quagga.optimizers.interruption import ValidationInterruption
from quagga.blocks import GrammarDataBlock, EmbeddingBlock, NpLstmRnn, MergeBlock, MeanPoolingBlock, LogisticRegressionCe


if __name__ == '__main__':
    embedding_init = lambda: np.random.rand(100, 2).astype(np.float32)
    W_init = lambda: np.random.rand(75, 100).astype(np.float32)
    R_init = lambda: np.random.rand(75, 75).astype(np.float32)
    log_reg_init = lambda: np.random.rand(1, 75).astype(np.float32)

    grammar_data_block = GrammarDataBlock(device_id=0)
    forward_embedding_block = EmbeddingBlock(embedding_init, grammar_data_block.data, device_id=0)
    backward_embedding_block = EmbeddingBlock(embedding_init, grammar_data_block.data, True, device_id=1)
    forward_rnn = NpLstmRnn(W_init, R_init, forward_embedding_block.output, device_id=0)
    backward_rnn = NpLstmRnn(W_init, R_init, backward_embedding_block.output, device_id=1)
    merge_block = MergeBlock(forward_rnn.h, backward_rnn.h, device_id=0)
    log_reg = LogisticRegressionCe(log_reg_init, merge_block.ouput, grammar_data_block.y, device_id=0)

    model = Model(grammar_data_block, forward_embedding_block, backward_embedding_block,
                  forward_rnn, backward_rnn, merge_block, log_reg)

    learning_rate_policy = FixedLearningRatePolicy(0.1)
    sgd = SgdOptimizer(20000, learning_rate_policy, model)
    sgd.add_interruption(ValidationInterruption(1000, 1000, 'test.log', model))
    sgd.optimize()
