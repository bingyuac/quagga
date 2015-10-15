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

    def get_last_modification_contexts(self):
        last_modification_contexts = []
        for column_indxs, v in self.columns.iteritems():
            last_modification_contexts.append(column_indxs.last_modification_context)
            for dense_matrix in v:
                last_modification_contexts.append(dense_matrix.last_modification_context)
        for row_indxs, v in self.rows.iteritems():
            last_modification_contexts.append(row_indxs.last_modification_context)
            for dense_matrix in v:
                last_modification_contexts.append(dense_matrix.last_modification_context)
        for rows_indxs, v in self.rows_batch.iteritems():
            last_modification_contexts.append(rows_indxs.last_modification_context)
            for dense_matrices in v:
                for dense_matrix in dense_matrices:
                    last_modification_contexts.append(dense_matrix.last_modification_context)
        return last_modification_contexts