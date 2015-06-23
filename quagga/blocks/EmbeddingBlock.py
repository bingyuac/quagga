from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class EmbeddingBlock(object):
    def __init__(self, embedding_init, indexes, device_id, reverse=False):
        self.device_id = device_id
        self.embedding = Matrix.from_npa(embedding_init(), device_id=device_id)
        self.context = Context(device_id)
        self.output = Connector(Matrix.empty(embedding_init.nrows, indexes.ncols, 'float', device_id),
                                self.context, self.context)
        self.indexes = indexes.register_usage(self.context)

    def fprop(self):
        self.output.ncols = self.indexes.ncols
        self.embedding.slice_columns(self.context, self.indexes, self.output)
        self.output.fprop()

    def bprop(self):
        # TODO
        pass

    @property
    def params(self):
        return [(self.context, self.embedding)]

    @property
    def grads(self):
        return [(self.context, self.embedding)]

    @property
    def f(self):
        return [{self.indexes: self.output.backward_matrix}]