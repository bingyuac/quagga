from collections import defaultdict
from quagga.matrix import MemoryBuffer


class SparseMatrix(object):
    def __init__(self, device_id):
        self.device_id = device_id
        self.columns = defaultdict(list)
        self.rows = defaultdict(list)
        self.rows_batch = defaultdict(list)
        self._memory_buffer = MemoryBuffer(device_id)

    def add_columns_slice(self, column_indxs, dense_matrix):
        self.columns[column_indxs].append(dense_matrix)

    def add_rows_slice(self, row_indxs, dense_matrix):
        self.rows[row_indxs].append(dense_matrix)

    def add_rows_batch_slice(self, rows_indxs, dense_matrices):
        self.rows_batch[rows_indxs].append(dense_matrices)

    def assign(self, context, a):
        """
        self <- a
        """

        self.clear()
        if self.device_id == a.device_id:
            # I avoided deepcopy because I do not know consequences
            # of deepcopy Matrix
            for k, v in a.columns.iteritems():
                self.columns[k] = [e for e in v]
            for k, v in a.rows.iteritems():
                self.rows[k] = [e for e in v]
            for k, v in a.rows_batch.iteritems():
                self.rows_batch[k] = [[m for m in ms] for ms in v]
        else:
            matrices = []
            for k, v in a.columns.iteritems():
                matrices.append(k)
                matrices.extend(v)
            for k, v in a.rows.iteritems():
                matrices.append(k)
                matrices.extend(v)
            for k, v in a.rows_batch.iteritems():
                matrices.append(k)
                matrices.extend(m for ms in v for m in ms)
            self._memory_buffer.extend_if_not_enough_space(matrices)

            for k, v in a.columns.iteritems():
                k = self._memory_buffer.get_matrix_copy(k)
                v = [self._memory_buffer.get_matrix_copy(context, e) for e in v]
                self.columns[k] = v
            for k, v in a.rows.iteritems():
                k = self._memory_buffer.get_matrix_copy(k)
                v = [self._memory_buffer.get_matrix_copy(context, e) for e in v]
                self.columns[k] = v
            for k, v in a.rows_batch.iteritems():
                k = self._memory_buffer.get_matrix_copy(k)
                for ms in v:
                    self.rows_batch[k].append([self._memory_buffer.get_matrix_copy(context, m) for m in ms])

    def clear(self):
        self.columns.clear()
        self.rows.clear()
        self.rows_batch.clear()
        self._memory_buffer.clear()