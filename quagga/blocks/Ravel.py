from quagga.context import Context
from quagga.connector import Connector


class Ravel(object):
    def __init__(self, matrix, device_id, propagate_error=True):
        self.context = Context(device_id)
        if propagate_error:
            self.matrix, self.dL_dmatrix = matrix.register_usage(self.context, self.context)
        else:
            self.matrix = matrix.register_usage(self.context)
        self.propagate_error = propagate_error
        self.output = Connector(self.matrix.ravel(), self.context, self.context)

    def fprop(self):
        self.output.nrows = self.matrix.nelems
        self.output.fprop()

    def bprop(self):
        if self.propagate_error:
            self.matrix.copy(self.context, self.dL_dmatrix)