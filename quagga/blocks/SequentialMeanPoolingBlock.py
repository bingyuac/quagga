import ctypes as ct
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class SequentialMeanPoolingBlock(object):
    def __init__(self, matrices, device_id=None):
        if all(m.bpropagable for m in matrices):
            learning = True
        elif all(not m.bpropagable for m in matrices):
            learning = False
        else:
            raise ValueError('All elements of matrices should be '
                             'bpropagable or non-bpropagable. '
                             'Mixed state is not allowed!')
        self.max_input_sequence_len = len(matrices)
        self.context = Context(device_id)
        self.output = Matrix.empty_like(matrices[0], self.context.device_id)
        self.output = Connector(self.output, self.context, self.context if learning else None)
        self._matrices = matrices
        self.matrices = []
        if learning:
            self.dL_dmatrices = []
        for matrix in matrices:
            if learning:
                matrix, dL_dmatrix = matrix.register_usage(self.context, self.context)
                self.dL_dmatrices.append(dL_dmatrix)
            else:
                matrix = matrix.register_usage(self.context)
            self.matrices.append(matrix)

    def fprop(self):
        n = len(self._matrices)
        if n > self.max_input_sequence_len:
            raise ValueError('Sequence has length: {} that is too long. '
                             'The maximum is: {}'.
                             format(n, self.max_input_sequence_len))
        self.output.assign_sequential_mean_pooling(self.context, self.matrices[:n])
        self.output.fprop()

    def bprop(self):
        n = len(self._matrices)
        dL_doutput = self.output.backward_matrix
        dL_doutput.scale(self.context, ct.c_float(1.0 / n))
        Matrix.sequentially_tile(self.context, self.dL_dmatrices, dL_doutput)