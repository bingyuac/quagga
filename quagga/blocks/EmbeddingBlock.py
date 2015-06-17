from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import Connector


class EmbeddingBlock(object):
    def __init__(self, embedding_init, indexes, indexes_max_len, reverse=False):
        self.embedding = Matrix.from_npa(embedding_init())
        self.buffer = Matrix.empty(embedding_init.nrows, indexes_max_len, 'float')
        self.output = Connector(self.buffer, Context())
        self.indexes = indexes
        self.indexes_max_len = indexes_max_len

    def fprop(self):
        self.output.forward_matrix = self.buffer[:, :self.indexes.nrows]
        self.indexes.forward_block(self.output)
        self.embedding.slice_columns(self.output.forward_context, self.indexes, self.output.forward_matrix)

    def bprop(self):
        pass

    @property
    def parameters(self):
        return [self.embedding]

    @property
    def f(self):
        return [{self.indexes: self.output.backward_matrix}]