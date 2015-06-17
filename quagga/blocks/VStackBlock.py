from itertools import izip
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import Connector


class VStackBlock(object):
    """
    VStackBlock concatenates matrices vertically. Can handle matrices with
    varying number of columns but only if it is the same across all matrices
    """

    def __init__(self, max_ncols, *matrices):
        dtype = matrices[0].dtype
        for matrix in matrices:
            if matrix.dtype != dtype:
                raise ValueError("Can't stack matrices with different dtypes!")
        self.matrices = matrices
        self.max_ncols = max_ncols
        nrows = sum(matrix.nrows for matrix in matrices)
        self.context = Context()
        self.output = Connector(Matrix.empty(nrows, max_ncols, dtype), self.context)

        self.dL_dbuffers = []
        for matrix in matrices:
            self.dL_dbuffers.append(Matrix.empty(matrix.nrows, max_ncols, dtype))
            matrix.register_user(self, self.context, self.dL_dbuffers[-1])

    def fprop(self):
        ncols = self.matrices[0].ncols
        for matrix in self.matrices:
            if matrix.ncols > self.max_ncols:
                raise ValueError('One of the matrix is too big!')
            if matrix.ncols != ncols:
                raise ValueError('VStackBlock concatenates matrices only when '
                                 'number of columns is the same across all '
                                 'matrices!')
        self.output.forward_matrix.ncols = ncols
        for matrix in self.matrices:
            matrix.block(self.context)
        self.output.forward_matrix.assign_vstack(self.context, self.matrices)

    def bprop(self, ):
        for matrix, dL_dbuffer in izip(self.matrices, self.dL_dbuffers):
            dL_dbuffer.ncols = matrix.ncols
        self.output.backward_block(self.context)
        self.output.backward_matrix.vsplit(self.context, self.dL_dbuffers)