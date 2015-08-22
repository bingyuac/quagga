from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class SequentialMeanPoolingBlock(object):
    def __init__(self, matrices, learning=True, device_id=None):
        self.max_input_sequence_len = len(matrices)
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.output = Matrix.empty_like(matrices, device_id)
        self.output = Connector(self.output, self.context, self.context if learning else None)
        self.matrices = []
        if learning:
            self.dL_dx = []
        for matrix in matrices:
            if learning:
                if not matrix.bpropagable:
                    raise ValueError('All elements of MatrixContainer should '
                                     'be bpropagable during learning!')
                matrix, dL_dx = matrix.register_usage(self.context, self.context)
                self.dL_dx.append(dL_dx)
            else:
                matrix = matrix.register_usage(self.context)
            self.matrices.append(matrix)
            self._matrices = matrices

    def fprop(self):
        self.output.assign_sequential_mean_pooling()
        self.output.fprop()

    def bprop(self):
        n = len(self._matrices)
        dL_doutput = self.output.backward_matrix
        dL_doutput.scale(self.context, self.alpha)
        if hasattr(self, 'dL_dmatrix'):
            self.dL_dmatrix.sequential_tile(self.context, self.axis, dL_doutput)