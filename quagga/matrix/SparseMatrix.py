from quagga.matrix import MemoryBuffer


class SparseMatrix(object):
    def __init__(self, device_id):
        self.device_id = device_id
        self.indexes = []
        self.dense_matrices = []
        self._memory_buffer = MemoryBuffer(device_id)

    def add(self, sparse_matrix):
        self.indexes.append(sparse_matrix.indexes)
        self.dense_matrices.append(sparse_matrix.dense_matrices)

    def copy_to(self, context, sparse_matrix):
        """
        self -> out
        """
        self._memory_buffer.extend_if_not_enough_space(sparse_matrix.indexes + sparse_matrix.dense_matrices)
        for indexes in sparse_matrix.indexes:
            self._memory_buffer.get_matrix_copy(context, indexes)
        for dense_matrix in sparse_matrix.dense_matrices:
            self._memory_buffer.get_matrix_copy(context, dense_matrix)

    def clear(self):
        self.indexes = []
        self.dense_matrices = []
        self._memory_buffer.clear()