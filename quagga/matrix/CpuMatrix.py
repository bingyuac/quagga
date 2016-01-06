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
import quagga
import weakref
import numpy as np
from itertools import izip
from quagga.matrix import ShapeElement


class CpuMatrix(object):
    def __init__(self, data, nrows, ncols, dtype):
        self.data = data
        self._nrows = nrows if isinstance(nrows, ShapeElement) else ShapeElement(nrows)
        self._ncols = ncols if isinstance(ncols, ShapeElement) else ShapeElement(ncols)
        self.dtype = dtype
        self.device_id = 0
        self.last_modification_context = None
        self.last_usage_context = None

    @staticmethod
    def get_setable_attributes():
        return ['nrows', 'ncols', 'npa']

    @property
    def npa(self):
        return self.data[:self.nrows.value, :self.ncols.value]

    @npa.setter
    def npa(self, value):
        self.data[:self.nrows.value, :self.ncols.value] = value

    @property
    def nelems(self):
        return self._nrows.value * self._ncols.value

    @property
    def nrows(self):
        return self._nrows

    @nrows.setter
    def nrows(self, value):
        data = self.data.base if self.data.base is not None else self.data
        if value > data.shape[0]:
            raise ValueError('There is no so many preallocated memory! '
                             'Maximum for `nrows` is {}'.format(self.data.shape[0]))
        self._nrows[:] = value

    @property
    def ncols(self):
        return self._ncols

    @ncols.setter
    def ncols(self, value):
        data = self.data.base if self.data.base is not None else self.data
        if value > data.shape[1]:
            raise ValueError('There is no so many preallocated memory! '
                             'Maximum for `ncols` is {}'.format(self.data.shape[1]))
        self._ncols[:] = value

    def __getitem__(self, key):
        # get row
        self_proxy = weakref.proxy(self)
        if isinstance(key, int):
            data = self.npa[key, np.newaxis]
            a = CpuMatrix(data, 1, self.ncols, self.dtype)
            a_proxy = weakref.proxy(a)
            if isinstance(self.ncols, ShapeElement):
                modif_handler = lambda: setattr(a_proxy, 'data', self_proxy.data[key, np.newaxis])
                self.ncols.add_modification_handler(modif_handler)
            return a
        if isinstance(key, ShapeElement):
            data = self.npa[key.value, np.newaxis]
            a = CpuMatrix(data, 1, self.ncols, self.dtype)
            a_proxy = weakref.proxy(a)
            modif_handler = lambda: setattr(a, 'data', self_proxy.data[key.value, np.newaxis])
            key.add_modification_handler(modif_handler)
            if isinstance(self.ncols, ShapeElement):
                modif_handler = lambda: setattr(a_proxy, 'data', self_proxy.data[key.value, np.newaxis])
                self.ncols.add_modification_handler(modif_handler)
            return a
        if isinstance(key, slice) and self.ncols == 1:
            key = (key, 0)
        # get row slice with one column
        if isinstance(key[0], slice) and not key[0].step and isinstance(key[1], (int, ShapeElement)):
            start = key[0].start if key[0].start else 0
            stop = key[0].stop if key[0].stop else self.nrows
            nrows = stop - start
            if isinstance(start, int) and isinstance(key[1], int):
                data = self.npa[start:, key[1], np.newaxis]
                a = CpuMatrix(data, nrows, 1, self.dtype)
                if isinstance(nrows, ShapeElement):
                    a_proxy = weakref.proxy(a)
                    modif_handler = lambda: setattr(a_proxy, 'data', self_proxy.data[start:, key[1], np.newaxis])
                    nrows.add_modification_handler(modif_handler)
                return a
            elif isinstance(start, int) and isinstance(key[1], ShapeElement):
                data = self.npa[start:, key[1].value, np.newaxis]
                a = CpuMatrix(data, nrows, 1, self.dtype)
                a_proxy = weakref.proxy(a)
                modif_handler = lambda: setattr(a_proxy, 'data', self_proxy.data[start:, key[1].value, np.newaxis])
                key[1].add_modification_handler(modif_handler)
                return a
            elif isinstance(start, ShapeElement) and isinstance(key[1], int):
                data = self.npa[start.value:, key[1], np.newaxis]
                a = CpuMatrix(data, nrows, 1, self.dtype)
                a_proxy = weakref.proxy(a)
                modif_handler = lambda: setattr(a_proxy, 'data', self_proxy.data[start.value:, key[1], np.newaxis])
                start.add_modification_handler(modif_handler)
                return a
            elif isinstance(start, ShapeElement) and isinstance(key[1], ShapeElement):
                data = self.npa[start.value:, key[1].value, np.newaxis]
                a = CpuMatrix(data, nrows, 1, self.dtype)
                a_proxy = weakref.proxy(a)
                modif_handler = lambda: setattr(a_proxy, 'data', self_proxy.data[start.value:, key[1].value, np.newaxis])
                key[1].add_modification_handler(modif_handler)
                start.add_modification_handler(modif_handler)
                return a
        # get column slice
        if key[0] == slice(None) and isinstance(key[1], slice) and not key[1].step:
            stop = key[1].stop if key[1].stop else self.ncols
            start = key[1].start if key[1].start else 0
            ncols = stop - start
            if isinstance(start, int):
                data = self.npa[:, start:]
                a = CpuMatrix(data, self.nrows, ncols, self.dtype)
                a_proxy = weakref.proxy(a)
                if isinstance(self.nrows, ShapeElement):
                    modif_handler = lambda: setattr(a_proxy, 'data', self_proxy.data[:, start:])
                    self.nrows.add_modification_handler(modif_handler)
                return a
            elif isinstance(start, ShapeElement):
                data = self.npa[:, start.value:]
                a = CpuMatrix(data, self.nrows, ncols, self.dtype)
                a_proxy = weakref.proxy(a)
                modif_handler = lambda: setattr(a_proxy, 'data', self_proxy.data[:, start.value:])
                start.add_modification_handler(modif_handler)
                if isinstance(self.nrows, ShapeElement):
                    modif_handler = lambda: setattr(a_proxy, 'data', self_proxy.data[:, start.value:])
                    self.nrows.add_modification_handler(modif_handler)
                return a
        raise ValueError('This slice: {} is unsupported!'.format(key))

    def same_shape(self, other):
        return self.npa.shape == other.npa.shape

    @staticmethod
    def str_to_dtype(dtype):
        if dtype == 'float':
            return np.float32
        if dtype == 'int':
            return np.int32
        raise TypeError(u'data type {} not understood'.format(dtype))

    @staticmethod
    def array_to_dtypes(a):
        if a.dtype == np.float32:
            return 'float', np.float32
        if a.dtype == np.int32:
            return 'int', np.int32
        raise TypeError(u'data type {} not understood'.format(a.dtype))

    @classmethod
    def from_npa(cls, a, dtype=None, device_id=None):
        if a.ndim != 2:
            raise ValueError('CpuMatrix works only with 2-d numpy arrays!')
        if dtype is not None:
            np_dtype = cls.str_to_dtype(dtype)
        else:
            dtype, np_dtype = cls.array_to_dtypes(a)
        if a.dtype != np_dtype:
            a = a.astype(dtype=np_dtype)
        return cls(np.copy(a), a.shape[0], a.shape[1], dtype)

    @classmethod
    def empty(cls, nrows, ncols, dtype=None, device_id=None):
        dtype = dtype if dtype else quagga.dtype
        np_dtype = cls.str_to_dtype(dtype)
        a = cls(None, nrows, ncols, dtype)
        nrows = nrows.value if isinstance(nrows, ShapeElement) else nrows
        ncols = ncols.value if isinstance(ncols, ShapeElement) else ncols
        a.data = np.nan_to_num(np.empty((nrows, ncols), dtype=np_dtype))
        return a

    @classmethod
    def empty_like(cls, other, device_id=None):
        return cls.empty(other.nrows, other.ncols, other.dtype)

    def to_host(self, context=None):
        return np.copy(self.npa)

    def assign(self, context, a):
        self.nrows, self.ncols = a.nrows, a.ncols
        self.npa = np.copy(a.npa)

    def assign_npa(self, context, a, nrows=None, ncols=None):
        # TODO(sergii): add support for ctypes pointer
        if self.npa.dtype != a.dtype:
            raise ValueError("Allocated memory has {} type. "
                             "Can't transfer to the device {} type".
                             format(self.npa.dtype, a.dtype))
        if a.ndim != 2:
            raise ValueError('CpuMatrix works only with 2-d numpy arrays!')
        self.nrows, self.ncols = a.shape
        self.npa = np.copy(a)

    def fill(self, context, value):
        self.npa = value

    def sync_fill(self, value):
        self.npa = value

    def slice_columns(self, context, column_indxs, out):
        out.npa = self.npa[:, column_indxs.npa.flatten()]

    def add_scaled_columns_slice(self, context, column_indxs, alpha, a):
        """
        self[:, column_indxs] += alpha * a
        """
        for i, idx in enumerate(column_indxs.npa.flatten()):
            self.npa[:, idx] += alpha * a.npa[:, i]

    def add_columns_slice(self, context, column_indxs, a):
        """
        self[:, column_indxs] += a
        """
        self.add_scaled_columns_slice(context, column_indxs, 1.0, a)

    def slice_columns_and_transpose(self, context, column_indxs, out):
        out.npa = self.npa[:, column_indxs.npa.flatten()].T

    def slice_rows(self, context, row_indxs, out):
        out.npa = self.npa[row_indxs.npa.flatten()]

    def add_scaled_rows_slice(self, context, row_indxs, alpha, a):
        """
        self[row_indxs] += alpha * a
        """
        for i, idx in enumerate(row_indxs.npa.flatten()):
            self.npa[idx] += alpha * a.npa[i]

    def add_rows_slice(self, context, row_indxs, a):
        """
        self[row_indxs] += a
        """
        self.add_scaled_rows_slice(context, row_indxs, 1.0, a)

    def slice_rows_batch(self, context, rows_indxs, dense_matrices):
        """
        for k in range(K):
            dense_matrices[k] = self[rows_indxs[:, k]]
        """
        n = rows_indxs.ncols
        for i in xrange(n):
            dense_matrices[i].npa = self.npa[rows_indxs.npa[:, i]]

    def add_scaled_rows_batch_slice(self, context, rows_indxs, alpha, dense_matrices):
        """
        for k in range(K):
            self[rows_indxs[:, k]] += alpha * dense_matrices[k]
        """
        for k, m in enumerate(dense_matrices):
            for i, idx in enumerate(rows_indxs.npa[:, k]):
                self.npa[idx] += alpha * m.npa[i]

    def add_rows_batch_slice(self, context, rows_indxs, dense_matrices):
        self.add_scaled_rows_batch_slice(context, rows_indxs, 1.0, dense_matrices)

    def assign_hstack(self, context, matrices):
        ncols = 0
        for matrix in matrices:
            ncols += int(matrix.ncols)
            if matrix.nrows != self.nrows:
                raise ValueError("The number of rows in the assigning matrix "
                                 "differs from the number of rows in buffers!")
        if ncols != self.ncols:
            raise ValueError("The number of columns in the assigning matrix differs"
                             "from the summed numbers of columns in buffers!")
        self.npa = np.hstack(m.npa for m in matrices)

    def hsplit(self, context, matrices, col_slices=None):
        if col_slices:
            for i, col_slice in enumerate(col_slices):
                matrices[i].npa = self.npa[:, col_slice[0]:col_slice[1]]
        else:
            ncols = 0
            for matrix in matrices:
                ncols += int(matrix.ncols)
                if matrix.nrows != self.nrows:
                    raise ValueError("The number of rows in the matrix to be split "
                                     "differs from the number of rows in buffers!")
            if ncols != self.ncols:
                raise ValueError("The number of columns in the matrix to be split differs "
                                 "from the summed numbers of columns in buffers!")
            indices_or_sections = [matrices[0].ncols]
            for m in matrices[1:-1]:
                indices_or_sections.append(indices_or_sections[-1] + m.ncols)
            _matrices = np.hsplit(self.npa, indices_or_sections)
            for _m, m in izip(_matrices, matrices):
                m.npa = _m

    @staticmethod
    def batch_hstack(context, x_sequence, y_sequence, output_sequence):
        for x, y, out in izip(x_sequence, y_sequence, output_sequence):
            out.npa = np.hstack((x.npa, y.npa))

    @staticmethod
    def batch_hsplit(context, input_sequence, x_sequence, y_sequence):
        x_ncols = x_sequence[0].npa.shape[1]
        for in_matrix, x, y in izip(input_sequence, x_sequence, y_sequence):
            x.npa = in_matrix.npa[:, :x_ncols]
            y.npa = in_matrix.npa[:, x_ncols:]

    def assign_vstack(self, context, matrices):
        nrows = 0
        for matrix in matrices:
            nrows += int(matrix.nrows)
            if matrix.ncols != self.ncols:
                raise ValueError("The number of columns in the assigning matrix "
                                 "differs from the number of columns in buffers!")
        if nrows != self.nrows:
            raise ValueError("The number of rows in the assigning matrix differs"
                             "from the summed numbers of rows in buffers!")
        self.npa = np.vstack(m.npa for m in matrices)

    def vsplit(self, context, matrices, row_slices=None):
        if row_slices:
            for i, row_slice in enumerate(row_slices):
                matrices[i].npa = self.npa[row_slice[0]:row_slice[1], :]
        else:
            nrows = 0
            for matrix in matrices:
                nrows += int(matrix.nrows)
                if matrix.ncols != self.ncols:
                    raise ValueError("The number of columns in the matrix to be split "
                                     "differs from the number of columns in buffers!")
            if nrows != self.nrows:
                raise ValueError("The number of rows in the matrix to be split differs "
                                 "from the summed numbers of rows in buffers!")
            indices_or_sections = [matrices[0].nrows]
            for m in matrices[1:-1]:
                indices_or_sections.append(indices_or_sections[-1] + m.nrows)
            _matrices = np.vsplit(self.npa, indices_or_sections)
            for _m, m in izip(_matrices, matrices):
                m.npa = _m

    def assign_sequential_mean_pooling(self, context, matrices):
        for i in xrange(matrices[0].nrows):
            self.npa[i] = np.mean([matrix.npa[i] for matrix in matrices], axis=0)

    def assign_sequential_sum_pooling(self, context, matrices):
        for i in xrange(matrices[0].nrows):
            self.npa[i] = np.sum([matrix.npa[i] for matrix in matrices], axis=0)

    @staticmethod
    def sequentially_tile(context, a, matrices):
        for m in matrices:
            m.npa = a.npa

    def tile(self, context, axis, a):
        n = int(self.nrows if axis == 0 else self.ncols)
        self.npa = np.repeat(a.npa, n, axis)

    def assign_repeat(self, context, a, repeats, axis):
        reps = [1, 1]
        reps[axis] = int(repeats)
        self.npa = np.tile(a.npa, reps)

    def add_repeat_derivative(self, context, a, repeats, axis):
        n = self.npa.shape[axis]
        if axis == 0:
            for i in xrange(repeats):
                self.npa += a.npa[i*n:(i+1)*n]
        elif axis == 1:
            for i in xrange(repeats):
                self.npa += a.npa[:, i*n:(i+1)*n]
        else:
            raise ValueError('TODO')

    @staticmethod
    def get_random_generator(seed):
        return np.random.RandomState(seed)

    def dropout(self, context, generator, dropout_prob, out):
        out.npa = generator.binomial(n=1, p=1-dropout_prob, size=self.npa.shape).astype(np.float32) * self.npa

    def add_gaussian_noise(self, context, generator, mean, std, out):
        out.npa = generator.normal(loc=mean, scale=std, size=self.npa.shape).astype(np.float32) + self.npa

    def assign_mask_zeros(self, context, a, b):
        """
        self = a .* (b != 0)
        """

        self.npa = a.npa * (b.npa != 0)

    def add_mask_zeros(self, context, a, b):
        """
        self += a .* (b != 0)
        """

        self.npa += a.npa * (b.npa != 0)

    def assign_masked_addition(self, context, mask, a, b):
        """
        self = mask .* a + (1 - mask) .* b
        """

        self.npa = mask.npa * a.npa + (1 - mask.npa) * b.npa

    def add_hprod_one_minus_mask(self, context, mask, a):
        """
        self += (1 - mask) .* a
        """

        self.npa += (1 - mask.npa) * a.npa

    def mask_column_numbers_row_wise(self, context, numbers):
        """
        self[i, j] = j < numbers[i]
        """
        for i in xrange(numbers.npa.shape[0]):
            self.npa[i] = np.arange(self.npa.shape[1]) < numbers.npa[i]

    def clip(self, context, min_value, max_value, out=None):
        if out is None:
            out = self
        out.npa = np.clip(self.npa, min_value, max_value)

    def tanh(self, context, tanh_matrix, derivative_matrix=None):
        np.tanh(self.npa, tanh_matrix.npa)
        if derivative_matrix:
            derivative_matrix.npa = 1.0 - tanh_matrix.npa ** 2

    def sigmoid(self, context, sigmoid_matrix, derivative_matrix=None):
        sigmoid_matrix.npa = 1.0 / (1.0 + np.exp(-self.npa))
        if derivative_matrix:
            derivative_matrix.npa = sigmoid_matrix.npa * (1.0 - sigmoid_matrix.npa)

    def tanh_sigm(self, context, tanh_sigm_matrix, derivative_matrix=None, axis=0):
        """
        This is a fancy function that is used during forward propagation into
        lstm cell. It calculates for the first 1/4 elements along the axis
        tanh function and sigmoid for the 3/4 remaining elements.
        """

        n = self.npa.shape[axis] / 4
        if axis == 0:
            tanh_npa = np.tanh(self.npa[:n])
            sigmoid_npa = 1.0 / (1.0 + np.exp(-self.npa[n:]))
            tanh_sigm_matrix.npa = np.vstack((tanh_npa, sigmoid_npa))
        elif axis == 1:
            tanh_npa = np.tanh(self.npa[:, :n])
            sigmoid_npa = 1.0 / (1.0 + np.exp(-self.npa[:, n:]))
            tanh_sigm_matrix.npa = np.hstack((tanh_npa, sigmoid_npa))
        else:
            raise ValueError('TODO')
        if derivative_matrix:
            tanh_der_npa = 1.0 - tanh_npa ** 2
            sigmoid_der_npa = sigmoid_npa * (1.0 - sigmoid_npa)
            f = np.hstack if axis else np.vstack
            derivative_matrix.npa = f((tanh_der_npa, sigmoid_der_npa))

    def relu(self, context, relu_matrix, derivative_matrix=None):
        relu_matrix.npa = np.maximum(self.npa, 0.0)
        if derivative_matrix:
            derivative_matrix.npa = (self.npa > 0).astype(np.float32)

    def softmax(self, context, softmax_matrix):
        maximums = np.max(self.npa, axis=1, keepdims=True)
        softmax_matrix.npa = self.npa - maximums
        np.exp(softmax_matrix.npa, softmax_matrix.npa)
        z = np.sum(softmax_matrix.npa, axis=1, keepdims=True)
        softmax_matrix.npa /= z

    def add_softmax_derivative(self, context, softmax_matrix, deriv_matrix):
        grad_x = softmax_matrix.npa * deriv_matrix.npa
        grad_x -= softmax_matrix.npa * grad_x.sum(axis=1, keepdims=True)
        self.npa += grad_x

    def assign_softmax_ce_derivative(self, context, probs, target_classes):
        self.npa = probs.npa / probs.npa.shape[0]
        self.npa[range(probs.nrows), target_classes.npa.flatten()] -= 1.0 / probs.npa.shape[0]

    def add_softmax_ce_derivative(self, context, probs, target_classes):
        temp = probs.npa / probs.npa.shape[0]
        temp[range(probs.nrows), target_classes.npa.flatten()] -= 1.0 / probs.npa.shape[0]
        self.npa += temp

    def scale(self, context, alpha, out=None):
        if out:
            out.npa = (self.npa * alpha)
        else:
            self.npa *= alpha

    def assign_scaled_addition(self, context, alpha, a, b):
        """
        self = alpha * (a + b)
        """
        self.npa = alpha * (a.npa + b.npa)

    def assign_add(self, context, a, b):
        self.assign_scaled_addition(context, 1.0, a, b)

    def assign_scaled_subtraction(self, context, alpha, a, b):
        """
        self = alpha * (a - b)
        """
        self.npa = alpha * (a.npa - b.npa)

    def add_scaled_subtraction(self, context, alpha, a, b):
        self.npa += alpha * (a.npa - b.npa)

    def assign_sub(self, context, a, b):
        self.assign_scaled_addition(context, 1.0, a, b)

    def add_scaled(self, context, alpha, a):
        """
        self += alpha * a
        """

        if isinstance(a, CpuMatrix):
            self.npa += alpha * a.npa
        elif isinstance(a, quagga.matrix.SparseMatrix):
            for column_indxs, v in a.columns.iteritems():
                for dense_matrix in v:
                    self.add_scaled_columns_slice(context, column_indxs, alpha, dense_matrix)
            for row_indxs, v in a.rows.iteritems():
                for dense_matrix in v:
                    self.add_scaled_rows_slice(context, row_indxs, alpha, dense_matrix)
            for rows_indxs, v in a.rows_batch.iteritems():
                for dense_matrices in v:
                    self.add_scaled_rows_batch_slice(context, rows_indxs, alpha, dense_matrices)
        else:
            raise ValueError('TODO')

    def add(self, context, a):
        self.add_scaled(context, 1.0, a)

    def sub(self, context, a):
        self.add_scaled(context, -1.0, a)

    def assign_sum(self, context, matrices):
        self.npa = 0.0
        self.add_sum(context, matrices)

    def add_sum(self, context, matrices):
        for m in matrices:
            self.npa += m.npa

    def hprod(self, context, a):
        """
        self = self .* a
        """
        self.add_hprod(context, self, a, alpha=0.0)

    def add_hprod(self, context, a, b, c=None, alpha=1.0):
        """
        self = a .* b + alpha * self        or
        self = a .* b .* c + alpha * self
        """
        if not c:
            self.npa = a.npa * b.npa + alpha * self.npa
        else:
            self.npa = a.npa * b.npa * c.npa + alpha * self.npa

    def add_scaled_hprod(self, context, a, b, alpha, beta):
        """
        self = alpha * self + beta * a .* b
        """
        self.npa = alpha * self.npa + beta * a.npa * b.npa

    def assign_hprod(self, context, a, b, c=None):
        """
        self = a .* b
        self = a .* b .* c  or
        """
        if not c:
            np.multiply(a.npa, b.npa, self.npa)
        else:
            self.npa = a.npa * b.npa * c.npa

    def assign_sum_hprod(self, context, a, b, c, d, e=None, f=None, g=None, h=None, i=None, j=None, k=None):
        """
        self = a .* b + c .* d                                   or
        self = a .* b .* c + d .* e                              or
        self = a .* b .* c + d .* e + f .* g + h .* i + j .* k
        """
        np.multiply(a.npa, b.npa, self.npa)
        if k is not None:
            self.npa *= c.npa
            self.npa += d.npa * e.npa
            self.npa += f.npa * g.npa
            self.npa += h.npa * i.npa
            self.npa += j.npa * k.npa
        elif e is not None:
            self.npa *= c.npa
            self.npa += d.npa * e.npa
        else:
            self.npa += c.npa * d.npa

    def assign_hprod_sum(self, context, a, b):
        """
        self = sum(a .* b, axis=1)
        """
        np.sum(a.npa * b.npa, axis=1, out=self.npa, keepdims=True)

    def add_scaled_div_sqrt(self, context, alpha, a, b, epsilon):
        """
        self += alpha * a ./ sqrt(b + epsilon)
        """
        self.npa += alpha * a.npa / np.sqrt(b.npa + epsilon)

    def assign_dot(self, context, a, b, matrix_operation_a='N', matrix_operation_b='N'):
        self.add_dot(context, a, b, matrix_operation_a, matrix_operation_b, beta=0.0)

    def add_dot(self, context, a, b, matrix_operation_a='N', matrix_operation_b='N', alpha=1.0, beta=1.0):
        """
        self = alpha * op(a) * b + beta * self
        """
        self.npa *= beta
        a = a.npa if matrix_operation_a == 'N' else a.npa.T
        b = b.npa if matrix_operation_b == 'N' else b.npa.T
        self.npa += alpha * np.dot(a, b)

    def argmax(self, context, out, axis=1):
        out.npa[:, 0] = np.argmax(self.npa, axis=axis)