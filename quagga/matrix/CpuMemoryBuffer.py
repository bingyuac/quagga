import numpy as np
from quagga.matrix import CpuMatrix


class CpuMemoryBuffer(object):
    def __init__(self, device_id):
        pass

    def extend_if_not_enough_space(self, matrices):
        pass

    def get_matrix_copy(self, context, matrix):
        return CpuMatrix(np.copy(matrix.data), int(matrix.nrows), int(matrix.ncols), matrix.dtype)

    def clear(self):
        pass