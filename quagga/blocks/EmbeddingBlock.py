from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import Connector


class EmbeddingBlock(object):
    def __init__(self, embedding_init, indexes, indexes_max_len):
        self.embedding = Matrix.from_npa(embedding_init())
        self.output_buffer = Matrix.empty(embedding_init.nrows, indexes_max_len, 'float')
        self.output = Connector(None, Context())
        self.indexes = indexes
        self.indexes_max_len = indexes_max_len

    def fprop(self):
        self.output.matrix = self.output_buffer[:, :self.indexes.nrows]
        self.indexes.block(self.output)
        self.embedding.slice_columns(self.output.forward_context, self.indexes, self.output.matrix)