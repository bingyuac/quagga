from itertools import izip
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class HorizontalStackBlock(object):
    """
    HorizontalStackBlock concatenates matrices horizontally.
    """

    def __init__(self, *matrices, **kwargs):
        self.context = Context(kwargs.get('device_id'))
        device_id = self.context.device_id
        self.matrices = []
        self.dL_dmatrices = []
        self.bpropagable = []
        for matrix in matrices:
            print
            self.bpropagable.append(matrix.bpropagable)
            if matrix.bpropagable:
                matrix, dL_dmatrix = matrix.register_usage(device_id, device_id)
                self.dL_dmatrices.append(dL_dmatrix)
            else:
                matrix = matrix.register_usage(device_id)
            self.matrices.append(matrix)
        ncols = sum(matrix.ncols for matrix in matrices)
        dtype = matrices[0].dtype
        bu_device_id = device_id if self.dL_dmatrices else None
        output = Matrix.empty(matrices[0].nrows, ncols, dtype)
        self.output = Connector(output, bu_device_id)

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