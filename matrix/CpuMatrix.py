import ctypes
import numpy as np


class CpuMatrix(object):
    def __init__(self, npa, nrows, ncols):
        self.npa = npa
        self.nrows = nrows
        self.ncols = ncols
        self.nelems = nrows * ncols

    def __getitem__(self, key):
        if type(key[1]) == int:
            # This is a workaround for slicing with np.newaxis
            # https://github.com/numpy/numpy/issues/5918
            # should be just:
            # key += (np.newaxis, )
            key = (key[0], slice(key[1], key[1] + 1, None))
        return CpuMatrix.from_npa(self.npa[key])

    @classmethod
    def from_npa(cls, a):
        if a.ndim != 2:
            raise ValueError('CpuMatrix works only with 2-d numpy arrays!')
        if not np.isfortran(a):
            a = np.asfortranarray(a, dtype=np.float32)
        elif a.dtype != np.float32:
            a = a.astype(dtype=np.float32)
        return cls(a, a.shape[0], a.shape[1])

    @classmethod
    def empty(cls, nrows, ncols):
        return cls.from_npa(np.nan_to_num(np.empty((nrows, ncols), dtype=np.float32)))

    @classmethod
    def empty_like(cls, other):
        if hasattr(other, 'npa'):
            return cls.from_npa(np.nan_to_num(np.empty_like(other.npa)))
        return cls.empty(other.nrows, other.ncols)

    def to_host(self):
        return self.npa

    def to_list(self):
        return [self[:, i] for i in xrange(self.ncols)]

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

    def sliced_add(self, context, a, column_indxs, alpha=1.0):
        """
        self[column_indxs] += alpha * a
        """
        for i, idx in enumerate(column_indxs):
            self.npa[:, idx] += alpha * a.npa[:, i]

    def add_hprod(self, context, a, b, alpha=1.0):
        """
        self = alpha * self + a .* b
        """
        self.npa *= alpha
        self.npa += a.npa * b.npa

    @staticmethod
    def hprod(context, out, a, b, c=None):
        """
        out = a .* b
        out = a .* b .* c  or
        """
        if not c:
            np.multiply(a.npa, b.npa, out.npa)
        else:
            np.multiply(a.npa, b.npa, out.npa)
            out.npa *= c.npa

    @staticmethod
    def sum_hprod(context, out, a, b, c, d, e=None, f=None, g=None, h=None, i=None, j=None, k=None):
        """
        out = a .* b + c .* d                                   or
        out = a .* b .* c + d .* e                              or
        out = a .* b .* c + d .* e + f .* g + h .* i + j .* k
        """
        np.multiply(a.npa, b.npa, out.npa)
        if k is not None:
            out.npa *= c.npa
            out.npa += d.npa * e.npa
            out.npa += f.npa * g.npa
            out.npa += h.npa * i.npa
            out.npa += j.npa * k.npa
        elif e is not None:
            out.npa *= c.npa
            out.npa += d.npa * e.npa
        else:
            out.npa += c.npa * d.npa

    def assign_dot(self, context, a, b, matrix_operation='N', alpha=1.0):
        self.add_dot(context, a, b, matrix_operation, alpha, 0.0)

    def add_dot(self, context, a, b, matrix_operation='N', alpha=1.0, beta=1.0):
        """
        self = alpha * op(a) * b + beta * self
        """
        self.npa *= beta
        self.npa += alpha * np.dot(a.npa if matrix_operation == 'N' else a.npa.T, b.npa)

    def vdot(self, context, a):
        return ctypes.c_float(np.vdot(self.npa, a.npa))