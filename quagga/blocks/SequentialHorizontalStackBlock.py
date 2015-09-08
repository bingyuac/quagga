from itertools import izip
from itertools import chain
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.matrix import MatrixList
from quagga.connector import Connector


class SequentialHorizontalStackBlock(object):
    def __init__(self, x_sequence, y_sequence, device_id=None):
        """
        TODO
        """
        if all(e.bpropagable for e in chain(x_sequence, y_sequence)):
            learning = True
        elif all(not e.bpropagable for e in chain(x_sequence, y_sequence)):
            learning = False
        else:
            raise ValueError('All elements should be bpropagable or '
                             'non-bpropagable. Mixed state is not allowed!')
        x_ncols = x_sequence[0].ncols
        y_ncols = y_sequence[0].ncols
        dtype = x_sequence[0].dtype
        for x, y in izip(x_sequence, y_sequence):
            if x.ncols != x_ncols or y.ncols != y_ncols:
                raise ValueError("All matrices in the sequence should have the same number of columns!")
            if x.nrows != y.nrows:
                raise ValueError("Can't stack matrices in sequence with different number of rows!")
            if x.dtype != dtype or y.dtype != dtype:
                raise ValueError("Can't stack matrices with different dtypes!")

        self.context = Context(device_id)
        device_id = self.context.device_id

        self._x_sequence = x_sequence
        self._y_sequence = y_sequence
        self.x_sequence = []
        self.y_sequence = []

        if learning:
            self.dL_dx_sequences = []
            self.dL_dy_sequences = []
            b_usage_context = self.context
        else:
            b_usage_context = None

        output_sequence = []
        for x, y in izip(x_sequence, y_sequence):
            if learning:
                x, dL_dx = x.register_usage(self.context, self.context)
                y, dL_dy = y.register_usage(self.context, self.context)
                self.dL_dx_sequences.append(dL_dx)
                self.dL_dy_sequences.append(dL_dy)
            else:
                x = x.register_usage(self.context)
                y = y.register_usage(self.context)
            self.x_sequence.append(x)
            self.y_sequence.append(y)
            output_sequence.append(Connector(Matrix.empty(x.nrows, x_ncols+y_ncols, dtype, device_id), self.context, b_usage_context))
        self.output_sequence = MatrixList(output_sequence)

    def fprop(self):
        n = len(self._x_sequence)
        if n != len(self._y_sequence):
            raise ValueError('TODO!')
        self.output_sequence.set_length(n)
        Matrix.batch_hstack(self.context, self.x_sequence[:n], self.y_sequence[:n], self.output_sequence)

    def bprop(self):
        if hasattr(self, 'dL_dx_sequences'):
            dL_doutput_sequence = [e.backward_matrix for e in self.output_sequence]
            n = len(dL_doutput_sequence)
            Matrix.batch_hsplit(self.context, dL_doutput_sequence, self.dL_dx_sequences[:n], self.dL_dy_sequences[:n])