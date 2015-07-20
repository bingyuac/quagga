from quagga.context import Context
from quagga.connector import Connector


class Ravel(object):
    def __init__(self, matrix, device_id=None):
        self.context = Context(device_id)
        if matrix._b_usage_context:
            self.matrix, self.dL_dmatrix = matrix.register_usage(self.context, self.context)
            self.bprop = lambda: self.output.backward_matrix.copy(self.context, self.dL_dmatrix)
            self.output = Connector(self.matrix.ravel(), self.context, self.context)
        else:
            self.matrix = matrix.register_usage(self.context)
            self.bprop = lambda: None
            self.output = Connector(self.matrix.ravel(), self.context)

    def fprop(self):
        self.output.nrows = self.matrix.nelems
        self.output.fprop()