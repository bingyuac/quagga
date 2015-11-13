import ctypes as ct
from itertools import izip
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class SequentialMeanPoolingBlock(object):
    def __init__(self, matrices, device_id=None):
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.output = Matrix.empty_like(matrices[0], device_id)
        learning = matrices[0].bpropagable
        self.output = Connector(self.output, device_id if learning else None)
        if learning:
            self.matrices, self.dL_dmatrices = izip(*matrices.register_usage(device_id, device_id))
        else:
            self.matrices = matrices.register_usage(self.context)
        self.length = matrices.length

    def fprop(self):
        self.output.assign_sequential_mean_pooling(self.context, self.matrices[:self.length])
        self.output.fprop()

    def bprop(self):
        dL_doutput = self.output.backward_matrix
        dL_doutput.scale(self.context, ct.c_float(1.0 / self.length))
        Matrix.sequentially_tile(self.context, dL_doutput, self.dL_dmatrices[:self.length])