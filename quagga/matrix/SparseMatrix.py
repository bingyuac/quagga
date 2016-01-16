# ----------------------------------------------------------------------------
# Copyright 2015 Grammarly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from collections import defaultdict


class SparseMatrix(object):
    """
    Stores mapping of indices to dense matrices
    (:class:`quagga.matrix.CpuMatrix` or :class:`quagga.matrix.GpuMatrix`)
    ``SparseMatrix`` instances can be added to other such instances or instances
    of :class:`quagga.matrix.CpuMatrix` or :class:`quagga.matrix.GpuMatrix`.

    """
    def __init__(self):
        self.columns = defaultdict(list)
        self.rows = defaultdict(list)
        self.rows_batch = defaultdict(list)

    def add_columns_slice(self, column_indxs, dense_matrix):
        """
        Adds ``dense_matrix`` elements to the corresponding ``column_indxs``
        slice.

        self[:, column_indxs] += dense_matrix

        Parameters
        ----------
        column_indxs : :class:`~quagga.matrix.CpuMatrix` or \
            :class:`~quagga.matrix.GpuMatrix`
            Column indices determine the slice to which the ``dense_matrix``
            should be added.
        dense_matrix : :class:`~quagga.matrix.CpuMatrix` \
            or :class:`~quagga.matrix.GpuMatrix`
            Dense matrix.
        """
        self.columns[column_indxs].append(dense_matrix)

    def add_rows_slice(self, row_indxs, dense_matrix):
        """
        Adds ``dense_matrix`` elements to the corresponding ``row_indxs``
        slice.

        self[row_indxs, :] += dense_matrix

        Parameters
        ----------
        row_indxs : :class:`~quagga.matrix.CpuMatrix` or \
            :class:`~quagga.matrix.GpuMatrix`
            Row indices determine the slice to which the ``dense_matrix``
            should be added.
        dense_matrix : :class:`~quagga.matrix.CpuMatrix` or \
        :class:`~quagga.matrix.GpuMatrix`
            Dense matrix.
        """
        self.rows[row_indxs].append(dense_matrix)

    def add_rows_batch_slice(self, rows_indxs, dense_matrices):
        """

        Parameters
        ----------
        rows_indxs : list of dense matrices (:class:`~quagga.matrix.CpuMatrix`\
            or :class:`~quagga.matrix.GpuMatrix`)
            ``rows_indxs[k]`` determines the slice to which the
            ``dense_matrices[k]`` should be added.
        dense_matrices : list of dense matrices \
        (:class:`~quagga.matrix.CpuMatrix` or :class:`~quagga.matrix.GpuMatrix`)
        """
        self.rows_batch[rows_indxs].append(dense_matrices)

    def add(self, sparse_matrix):
        """
        Performs in-place addition of ``sparse_matrix``.

        Parameters
        ----------
        sparse_matrix : :class:`~quagga.matrix.SparseMatrix`
        """
        for k, v in sparse_matrix.columns.iteritems():
            self.columns[k].extend(v)
        for k, v in sparse_matrix.rows.iteritems():
            self.rows[k].extend(v)
        for k, v in sparse_matrix.rows_batch.iteritems():
            self.rows_batch[k].extend([m for m in ms] for ms in v)

    def clear(self):
        """
        Clears sparse matrix.
        """
        self.columns.clear()
        self.rows.clear()
        self.rows_batch.clear()

    @property
    def last_modif_contexts(self):
        """
        Returns all last modification contexts of dense matrices that are
        contained in the sparse matrix.

        Returns
        -------
        last_modif_contexts : list of :class:`~quagga.context.CpuContext` \
        or :class:`~quagga.context.GpuContext`
        """
        last_modif_contexts = []
        for column_indxs, v in self.columns.iteritems():
            last_modif_contexts.append(column_indxs.last_modif_context)
            for dense_matrix in v:
                last_modif_contexts.append(dense_matrix.last_modif_context)
        for row_indxs, v in self.rows.iteritems():
            last_modif_contexts.append(row_indxs.last_modif_context)
            for dense_matrix in v:
                last_modif_contexts.append(dense_matrix.last_modif_context)
        for rows_indxs, v in self.rows_batch.iteritems():
            last_modif_contexts.append(rows_indxs.last_modif_context)
            for dense_matrices in v:
                for dense_matrix in dense_matrices:
                    last_modif_contexts.append(dense_matrix.last_modif_context)
        return last_modif_contexts