from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class EmbeddingBlock(object):
    def __init__(self, embedding_init, indexes, device_id=None):
        self.embedding = Matrix.from_npa(embedding_init(), device_id=device_id)
        self.context = Context(device_id)
        self.output = Connector(Matrix.empty(self.embedding.nrows, indexes.ncols, 'float', device_id),
                                self.context, self.context)
        self.indexes = indexes.register_usage(self.context)
        self.reverse = reverse

    def fprop(self):
        self.output.ncols = self.indexes.ncols
        self.embedding.slice_columns(self.context, self.indexes, self.output, self.reverse)
        self.output.fprop()

    def bprop(self):
        self.output.bprop()

    @property
    def params(self):
        return [(self.context, self.embedding)]

    @property
    def grads(self):
        return [(self.context, (self.indexes, self.output.backward_matrix))]