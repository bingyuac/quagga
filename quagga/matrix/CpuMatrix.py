import numpy as np
import ctypes as ct


class CpuMatrix(object):
    def __init__(self, npa, nrows, ncols, dtype):
        self.npa = npa
        self.nrows = nrows
        self.ncols = ncols
        self.nelems = nrows * ncols
        self.dtype = dtype

    def __getitem__(self, key):
        if type(key[1]) == int:
            # This is a workaround for slicing with np.newaxis
            # https://github.com/numpy/numpy/issues/5918
            # should be just:
            # key += (np.newaxis, )
            key = (key[0], slice(key[1], key[1] + 1, None))
        return CpuMatrix.from_npa(self.npa[key])

    def slice_columns(self, context, column_indxs, out):
        out.npa = self.npa[:, column_indxs.npa.flatten()]

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
    def from_npa(cls, a, dtype=None):
        if a.ndim != 2:
            raise ValueError('CpuMatrix works only with 2-d numpy arrays!')
        dtype = cls.str_to_dtype(dtype) if dtype else a.dtype
        if not np.isfortran(a):
            a = np.asfortranarray(a, dtype=dtype)
        elif a.dtype != dtype:
            a = a.astype(dtype=dtype)
        return cls(a, a.shape[0], a.shape[1], dtype)

    @classmethod
    def empty(cls, nrows, ncols, dtype):
        np_dtype = cls.str_to_dtype(dtype) if type(dtype) is str else dtype
        return cls.from_npa(np.nan_to_num(np.empty((nrows, ncols), dtype=np_dtype)))

    @classmethod
    def empty_like(cls, other):
        if hasattr(other, 'npa'):
            return cls.from_npa(np.nan_to_num(np.empty_like(other.npa)))
        return cls.empty(other.nrows, other.ncols, other.dtype)

    def to_device(self, context, a):
        if self.npa.dtype != a.dtype:
            raise ValueError("Allocated memory has {} type. "
                             "Can't transfer to the device {} type".
                             format(self.npa.dtype, a.dtype))
        if a.ndim != 2:
            raise ValueError('GpuMatrix works only with 2-d numpy arrays!')
        if not np.isfortran(a):
            a = np.asfortranarray(a)
        self.nrows, self.ncols = a.shape
        self.nelems = self.nrows * self.ncols
        self.npa = a

    def to_host(self):
        return self.npa

    def to_list(self):
        return [self[:, i] for i in xrange(self.ncols)]

    def copy(self, context, out):
        out.npa.data = np.copy(out.npa).data

    def scale(self, context, alpha, out=None):
        if out:
            out.npa.data = (self.npa * alpha).data
        else:
            self.npa *= alpha

    def tanh(self, context, tanh_matrix, derivative_matrix=None):
        np.tanh(self.npa, tanh_matrix.npa)
        if derivative_matrix:
            derivative_matrix.npa.data = (1.0 - tanh_matrix.npa ** 2).data

    def sigmoid(self, context, sigmoid_matrix, derivative_matrix=None):
        sigmoid_matrix.npa.data = (1.0 / (1.0 + np.exp(-self.npa))).data
        if derivative_matrix:
            derivative_matrix.npa.data = (sigmoid_matrix.npa * (1.0 - sigmoid_matrix.npa)).data

    def add_scaled(self, context, alpha, a):
        """
        self += alpha * a
        """
        self.npa += alpha * a.npa

    def add(self, context, a, b=None, c=None):
        if not b and not c:
            self.add_scaled(context, 1.0, a)
        else:
            for m in [a, b, c]:
                self.npa += m.npa

    def sliced_add_scaled(self, context, column_indxs, alpha, a):
        """
        self[column_indxs] += alpha * a
        """
        for i, idx in enumerate(column_indxs.npa.flatten()):
            self.npa[:, idx] += alpha * a.npa[:, i]

    def add_hprod(self, context, a, b, alpha=1.0):
        """
        self = alpha * self + a .* b
        """
        self.npa *= alpha
        self.npa += a.npa * b.npa

    def assign_hprod(self, context, a, b, c=None):
        """
        self = a .* b
        self = a .* b .* c  or
        """
        if not c:
            np.multiply(a.npa, b.npa, self.npa)
        else:
            np.multiply(a.npa, b.npa, self.npa)
            self.npa *= c.npa

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