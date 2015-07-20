from itertools import izip
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class VerticalStackBlock(object):
    """
    VStackBlock concatenates matrices vertically. Can handle matrices with
    varying number of columns but only if it is the same across all matrices
    during `fprop` step. Number of rows is fixed.
    """

    def __init__(self, *matrices, **kwargs):
        dtype = matrices[0].dtype
        for matrix in matrices:
            if matrix.dtype != dtype:
                raise ValueError("Can't stack matrices with different dtypes!")
        self.max_ncols = matrices[0].ncols
        self.context = Context(kwargs.get('device_id'))
        self.matrices = []
        self.dL_dmatrices = []
        row_slices = []
        nrows = [0]
        for k, matrix in enumerate(matrices):
            nrows.append(nrows[-1] + matrix.nrows)
            if matrix._b_usage_context:
                matrix, dL_dmatrix = matrix.register_usage(self.context, self.context)
                self.dL_dmatrices.append(dL_dmatrix)
                row_slices.append((nrows[-2], nrows[-1]))
            else:
                matrix = matrix.register_usage(self.context)
            self.matrices.append(matrix)

        nrows = sum(matrix.nrows for matrix in matrices)
        if self.dL_dmatrices:
            self.output = Connector(Matrix.empty(nrows, self.max_ncols, dtype), self.context, self.context)
            self.bprop = lambda: self.output.backward_matrix.vsplit(self.context, self.dL_dmatrices, row_slices)
        else:
            self.output = Connector(Matrix.empty(nrows, self.max_ncols, dtype), self.context)
            self.bprop = lambda: None

    def fprop(self):
        self.output.ncols = self.matrices[0].ncols
        self.output.assign_vstack(self.context, self.matrices)
        self.output.fprop()
