from collections import defaultdict


class SparseMatrix(object):
    def __init__(self, device_id):
        self.device_id = device_id
        self.columns = defaultdict(list)
        self.rows = defaultdict(list)
        self.rows_batch = defaultdict(list)

    def add_columns_slice(self, column_indxs, dense_matrix):
        self.columns[column_indxs].append(dense_matrix)

    def add_rows_slice(self, row_indxs, dense_matrix):
        self.rows[row_indxs].append(dense_matrix)

    def add_rows_batch_slice(self, rows_indxs, dense_matrices):
        self.rows_batch[rows_indxs].append(dense_matrices)

    def add(self, sparse_matrix):
        for k, v in sparse_matrix.columns.iteritems():
            self.columns[k].extend(v)
        for k, v in sparse_matrix.rows.iteritems():
            self.rows[k].extend(v)
        for k, v in sparse_matrix.rows_batch.iteritems():
            self.rows_batch[k].extend([m for m in ms] for ms in v)

    def clear(self):
        self.columns.clear()
        self.rows.clear()
        self.rows_batch.clear()