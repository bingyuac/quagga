import quagga
import numpy as np
import ctypes as ct
from itertools import izip


class CpuMatrix(object):
    def __init__(self, npa, nrows, ncols, dtype, device_id):
        self.npa = npa
        self._nrows = nrows
        self._ncols = ncols
        self.dtype = dtype
        self.device_id = device_id

    @property
    def nelems(self):
        return self._nrows * self._ncols

    @property
    def nrows(self):
        return self._nrows

    @nrows.setter
    def nrows(self, value):
        base = self.npa.base
        if base is None:
            base = self.npa
        if value > base.shape[0]:
            raise ValueError('There is no so many preallocated memory! '
                             'Maximum for `nrows` is {}'.format(base.shape[0]))
        self._nrows = value
        self.npa = base[:value]
        if self.npa.base is None:
            raise ValueError('Oops!')

    @property
    def ncols(self):
        return self._ncols

    @ncols.setter
    def ncols(self, value):
        base = self.npa.base
        if base is None:
            base = self.npa
        if value > base.shape[1]:
            raise ValueError('There is no so many preallocated memory! '
                             'Maximum for `ncols` is {}'.format(base.shape[1]))
        self._ncols = value
        self.npa = base[:, :value]
        if self.npa.base is None:
            raise ValueError('Oops!')

    def __getitem__(self, key):
        if type(key[1]) == int:
            # This is a workaround for slicing with np.newaxis
            # https://github.com/numpy/numpy/issues/5918
            # should be just:
            # key += (np.newaxis, )
            key = (key[0], slice(key[1], key[1] + 1, None))
        return CpuMatrix.from_npa(self.npa[key])

    def same_shape(self, other):
        return self.nrows == other.nrows and self.ncols == other.ncols

    @staticmethod
    def str_to_dtype(dtype):
        if dtype == 'float':
            return np.float32
        if dtype == 'int':
            return np.int32
        raise TypeError(u'data type {} not understood'.format(dtype))

    @classmethod
    def from_npa(cls, a, dtype=None, device_id=None):
        if a.ndim != 2:
            raise ValueError('CpuMatrix works only with 2-d numpy arrays!')
        dtype = cls.str_to_dtype(dtype) if dtype else a.dtype
        if a.dtype != dtype:
            a = a.astype(dtype=dtype)
        return cls(a, a.shape[0], a.shape[1], dtype, device_id)

    @classmethod
    def empty(cls, nrows, ncols, dtype=None, device_id=None):
        dtype = dtype if dtype else quagga.dtype
        np_dtype = cls.str_to_dtype(dtype) if type(dtype) is str else dtype
        return cls.from_npa(np.nan_to_num(np.empty((nrows, ncols), dtype=np_dtype)), device_id=device_id)

    @classmethod
    def empty_like(cls, other, device_id=None):
        if hasattr(other, 'npa'):
            return cls.from_npa(np.nan_to_num(np.empty_like(other.npa)))
        return cls.empty(other.nrows, other.ncols, other.dtype, device_id)

    def to_device(self, context, a):
        if self.npa.dtype != a.dtype:
            raise ValueError("Allocated memory has {} type. "
                             "Can't transfer to the device {} type".
                             format(self.npa.dtype, a.dtype))
        if a.ndim != 2:
            raise ValueError('GpuMatrix works only with 2-d numpy arrays!')
        self.nrows, self.ncols = a.shape
        self.npa = a

    def fill(self, context, value):
        self.npa[...] = value

    def sync_fill(self, value):
        self.npa[...] = value

    def to_host(self):
        return np.copy(self.npa)

    def to_list(self):
        return [self[:, i] for i in xrange(self.ncols)]

    def copy(self, context, out):
        out.npa[...] = np.copy(self.npa)

    def tile(self, context, axis, a):
        n = self.nrows if axis == 0 else self.ncols
        self.npa[...] = np.repeat(a.npa, n, axis)

    def slice_columns(self, context, column_indxs, out):
        out.npa[...] = self.npa[:, column_indxs.npa.flatten()]

    def slice_columns_and_transpose(self, context, column_indxs, out):
        out.npa[...] = self.npa[:, column_indxs.npa.flatten()].T

    def slice_rows(self, context, row_indxs, out):
        out.npa[...] = self.npa[row_indxs.npa.flatten()]

    def assign_hstack(self, context, matrices):
        ncols = 0
        for matrix in matrices:
            ncols += matrix.ncols
            if matrix.nrows != self.nrows:
                raise ValueError("The number of rows in the assigning matrix "
                                 "differs from the number of rows in buffers!")
        if ncols != self.ncols:
            raise ValueError("The number of columns in the assigning matrix differs"
                             "from the summed numbers of columns in buffers!")
        self.npa[...] = np.hstack(m.npa for m in matrices)

    def hsplit(self, context, matrices, col_slices=None):
        if col_slices:
            for i, col_slice in enumerate(col_slices):
                matrices[i].npa[...] = self.npa[:, col_slice[0]:col_slice[1]]
        else:
            ncols = 0
            for matrix in matrices:
                ncols += matrix.ncols
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
                m.npa[...] = _m

    def assign_vstack(self, context, matrices):
        nrows = 0
        for matrix in matrices:
            nrows += matrix.nrows
            if matrix.ncols != self.ncols:
                raise ValueError("The number of columns in the assigning matrix "
                                 "differs from the number of columns in buffers!")
        if nrows != self.nrows:
            raise ValueError("The number of rows in the assigning matrix differs"
                             "from the summed numbers of rows in buffers!")
        self.npa[...] = np.vstack(m.npa for m in matrices)

    def vsplit(self, context, matrices, row_slices=None):
        if row_slices:
            for i, row_slice in enumerate(row_slices):
                matrices[i].npa[...] = self.npa[row_slice[0]:row_slice[1], :]
        else:
            nrows = 0
            for matrix in matrices:
                nrows += matrix.nrows
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
                m.npa[...] = _m

    def scale(self, context, alpha, out=None):
        if out:
            out.npa[...] = (self.npa * alpha)
        else:
            self.npa *= alpha

    def tanh(self, context, tanh_matrix, derivative_matrix=None):
        np.tanh(self.npa, tanh_matrix.npa)
        if derivative_matrix:
            derivative_matrix.npa[...] = 1.0 - tanh_matrix.npa ** 2

    def sigmoid(self, context, sigmoid_matrix, derivative_matrix=None):
        sigmoid_matrix.npa[...] = 1.0 / (1.0 + np.exp(-self.npa))
        if derivative_matrix:
            derivative_matrix.npa[...] = sigmoid_matrix.npa * (1.0 - sigmoid_matrix.npa)

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
            tanh_sigm_matrix.npa[...] = np.vstack((tanh_npa, sigmoid_npa))
        elif axis == 1:
            tanh_npa = np.tanh(self.npa[:, :n])
            sigmoid_npa = 1.0 / (1.0 + np.exp(-self.npa[:, n:]))
            tanh_sigm_matrix.npa[...] = np.hstack((tanh_npa, sigmoid_npa))
        else:
            raise ValueError('TODO')
        if derivative_matrix:
            tanh_der_npa = 1.0 - tanh_npa ** 2
            sigmoid_der_npa = sigmoid_npa * (1.0 - sigmoid_npa)
            f = np.hstack if axis else np.vstack
            derivative_matrix.npa[...] = f((tanh_der_npa, sigmoid_der_npa))

    def relu(self, context, relu_matrix, derivative_matrix=None):
        relu_matrix.npa[...] = np.maximum(self.npa, 0.0)
        if derivative_matrix:
            derivative_matrix.npa[...] = (self.npa > 0).astype(np.float32)

    def softmax(self, context, softmax_matrix):
        maximums = np.max(self.npa, axis=1, keepdims=True)
        softmax_matrix.npa[...] = self.npa - maximums
        np.exp(softmax_matrix.npa, softmax_matrix.npa)
        z = np.sum(softmax_matrix.npa, axis=1, keepdims=True)
        softmax_matrix.npa[...] /= z

    def assign_scaled_addition(self, context, alpha, a, b):
        """
        self = alpha * (a + b)
        """
        self.npa[...] = alpha * (a.npa + b.npa)

    def assign_add(self, context, a, b):
        self.assign_scaled_addition(context, 1.0, a, b)

    def assign_scaled_subtraction(self, context, alpha, a, b):
        """
        self = alpha * (a - b)
        """
        self.npa[...] = alpha * (a.npa - b.npa)

    def assign_softmax_ce_derivative(self, context, probs, target_classes):
        self.npa[...] = probs.npa / probs.npa.shape[0]
        self.npa[range(probs.npa.shape[0]), target_classes.npa.flatten()] -= 1.0 / probs.npa.shape[0]

    def assign_sub(self, context, a, b):
        self.assign_scaled_addition(context, 1.0, a, b)

    def add_scaled(self, context, alpha, a):
        """
        self += alpha * a
        """
        self.npa += alpha * a.npa

    def add(self, context, a):
        self.add_scaled(context, 1.0, a)

    def add_sum(self, context, matrices):
        for m in matrices:
            self.npa += m.npa

    def assign_sum(self, context, matrices):
        self.npa = 0.0
        self.add_sum(context, matrices)

    def sub(self, context, a):
        self.add_scaled(context, -1.0, a)

    def sliced_columns_add_scaled(self, context, column_indxs, alpha, a):
        """
        self[column_indxs] += alpha * a
        """
        for i, idx in enumerate(column_indxs.npa.flatten()):
            self.npa[:, idx] += alpha * a.npa[:, i]

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
            self.npa[...] = a.npa * b.npa + alpha * self.npa
        else:
            self.npa[...] = a.npa * b.npa * c.npa + alpha * self.npa

    def assign_hprod(self, context, a, b, c=None):
        """
        self = a .* b
        self = a .* b .* c  or
        """
        if not c:
            np.multiply(a.npa, b.npa, self.npa)
        else:
            self.npa[...] = a.npa * b.npa * c.npa

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

    def vdot(self, context, a):
        return ct.c_float(np.vdot(self.npa, a.npa))

    def assign_sequential_mean_pooling(self, context, matrices):
        for i in xrange(matrices[0].nrows):
            self.npa[i] = np.mean([matrix.npa[i] for matrix in matrices], axis=0)

    @staticmethod
    def sequentially_tile(context, matrices, a):
        for m in matrices:
            m.npa[...] = a.npa

    def slice_rows_batch(self, context, embd_rows_indxs, dense_matrices):
        n = embd_rows_indxs.ncols
        for i in xrange(n):
            dense_matrices[i].npa[...] = self.npa[embd_rows_indxs.npa[:, i]]

    def sliced_rows_batch_scaled_add(self, context, embd_rows_indxs, alpha, dense_matrices):
        """
        for k in range(K):
            self[column_indxs[:, k]] += alpha * dense_matrices[k]
        """
        for k, m in enumerate(dense_matrices):
            for i, idx in enumerate(embd_rows_indxs.npa[:, k]):
                self.npa[idx] += alpha * m.npa[i]

    @staticmethod
    def get_random_generator(seed):
        return np.random.RandomState(seed)

    def dropout(self, context, generator, dropout_prob, out):
        out.npa[...] = generator.binomial(n=1, p=1-dropout_prob, size=self.npa.shape) * self.npa

    def mask_zeros(self, context, mask, out):
        """
        out = self * (mask != 0)
        """
        out.npa[...] = self.npa * (mask.npa != 0)