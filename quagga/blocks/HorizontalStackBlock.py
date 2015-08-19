from itertools import izip
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class HorizontalStackBlock(object):
    """
    HorizontalStackBlock concatenates matrices horizontally.
    Can handle matrices with varying number of columns
    """

    def __init__(self, *matrices, **kwargs):
        dtype = matrices[0].dtype
        for matrix in matrices:
            if matrix.dtype != dtype:
                raise ValueError("Can't stack matrices with different dtypes!")
        self.context = Context(kwargs.get('device_id'))
        self.matrices = []
        self.dL_dmatrices = []
        self.bpropagable = []
        for matrix in matrices:
            if matrix.bpropagable:
                matrix, dL_dmatrix = matrix.register_usage(self.context, self.context)
                self.dL_dmatrices.append(dL_dmatrix)
                self.bpropagable.append(True)
            else:
                matrix = matrix.register_usage(self.context)
                self.bpropagable.append(False)
            self.matrices.append(matrix)
        ncols = sum(matrix.ncols for matrix in matrices)
        b_usage_context = self.context if self.dL_dmatrices else None
        self.output = Connector(Matrix.empty(matrices[0].nrows, ncols, dtype), self.context, b_usage_context)

    def fprop(self):
        self.output.assign_hstack(self.context, self.matrices)
        self.output.fprop()

    def bprop(self):
        if self.dL_dmatrices:
            col_slices = []
            ncols = [0]
            for matrix, bpropagable in izip(self.matrices, self.bpropagable):
                ncols.append(ncols[-1] + matrix.ncols)
                if bpropagable:
                    col_slices.append((ncols[-2], ncols[-1]))
            self.output.backward_matrix.hsplit(self.context, self.dL_dmatrices, col_slices)