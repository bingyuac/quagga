from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import Connector


class EmbeddingBlock(object):
    def __init__(self, embedding_init, indexes, indexes_max_len, device_id, reverse=False):
        self.device_id = device_id
        self.embedding = Matrix.from_npa(embedding_init(), device_id=device_id)
        self.context = Context(device_id)
        self.output = Connector(Matrix.empty(embedding_init.nrows, indexes_max_len, 'float', device_id),
                                self.context)
        self.indexes = indexes
        self.indexes.register_user(self, self.context)
        self.indexes_max_len = indexes_max_len

    def fprop(self):
        self.output.ncols = self.indexes.ncols
        self.embedding.slice_columns(self.output._f_obtaining_context, self.indexes, self.output.forward_matrix)

    def bprop(self):
        pass

    @property
    def parameters(self):
        return [self.embedding]

    @property
    def f(self):
        return [{self.indexes: self.output.backward_matrix}]