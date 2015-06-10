from quagga.matrix import Matrix
from quagga.blocks import Connector


class MergeBlock(object):
    def __init__(self, first_matrix, second_matrix, max_ncols):
        if first_matrix.nrows != second_matrix.nrows:
            raise ValueError("Can't horizontally stack matrices with different nrows!")
        if first_matrix.dtype != second_matrix.dtype:
            raise ValueError("Can't concatenate matrices with different dtypes!")

        self.merged_matrix = Connector(Matrix.empty(first_matrix.nrows, max_ncols, first_matrix.dtype))

    def fprop(self):
        self.merged_matrix()

    def bprop(self):
        pass