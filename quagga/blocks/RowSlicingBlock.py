from quagga.utils import List
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class RowSlicingBlock(object):
    def __init__(self, W, row_indexes):
        device_id = W.device_id
        self.context = Context(device_id)
        learning = W.bpropagable
        if learning:
            self.W, self.dL_dW = W.register_usage_with_sparse_backward_matrix()
        else:
            self.W = W.register_usage(device_id)
        self.row_indexes = row_indexes.register_usage(device_id)
        if row_indexes.ncols > 1:
            self.output = []
            for i in xrange(row_indexes.ncols):
                output = Matrix.empty(row_indexes.nrows, W.ncols, device_id=device_id)
                output = Connector(output, device_id if learning else None)
                self.output.append(output)
            self.output = List(self.output, row_indexes.ncols)
        else:
            output = Matrix.empty(row_indexes.nrows, W.ncols, device_id=device_id)
            self.output = Connector(output, device_id if learning else None)

    def fprop(self):
        if isinstance(self.output, List):
            self.W.slice_rows_batch(self.context, self.row_indexes, self.output)
        else:
            self.W.slice_rows(self.context, self.row_indexes, self.output)
        self.output.fprop()

    def bprop(self):
        if isinstance(self.output, List):
            self.dL_dW.add_rows_batch_slice(self.row_indexes, self.output.bprop())
        else:
            self.dL_dW.add_rows_slice(self.row_indexes, self.output.bprop())