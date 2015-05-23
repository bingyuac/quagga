import atexit
import ctypes
import numpy as np
from cuda import cudart, cublas, gpu_matrix_kernels


class GpuMatrix(object):
    def __init__(self, data, nrows, ncols, is_owner):
        self.data = data
        self.nrows = nrows
        self.ncols = ncols
        self.nelems = nrows * ncols
        self.nbytes = self.nelems * ctypes.sizeof(ctypes.c_float)
        self.is_owner = is_owner
        self.released = False
        if is_owner:
            atexit.register(lambda: None if self.released else cudart.cuda_free(self.data))

    def __getitem__(self, key):
        void_p = ctypes.cast(self.data, ctypes.c_voidp).value + self.nrows * key[1] * ctypes.sizeof(ctypes.c_float)
        data = ctypes.cast(void_p, ctypes.POINTER(ctypes.c_float))
        return GpuMatrix.from_device_array(data, self.nrows, 1)

    def __del__(self):
        print 'del {}'.format(self)
        if self.is_owner:
            print 'del released {}'.format(self)
            cudart.cuda_free(self.data)
            self.released = True

    @classmethod
    def from_npa(cls, a):
        if a.ndim != 2:
            raise ValueError('GpuMatrix works only with 2-d numpy arrays!')
        if not np.isfortran(a):
            a = np.asfortranarray(a, dtype=np.float32)
        elif a.dtype != np.float32:
            a = a.astype(dtype=np.float32)
        host_data = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        elem_size = ctypes.sizeof(ctypes.c_float)
        nbytes = a.size * elem_size
        data = cudart.cuda_malloc(nbytes, ctypes.c_float)
        cudart.cuda_memcpy(data, host_data, nbytes, 'host_to_device')
        return cls(data, a.shape[0], a.shape[1], True)

    @classmethod
    def empty(cls, nrows, ncols):
        nbytes = nrows * ncols * ctypes.sizeof(ctypes.c_float)
        data = cudart.cuda_malloc(nbytes, ctypes.c_float)
        return cls(data, nrows, ncols, True)

    @classmethod
    def empty_like(cls, other):
        nbytes = other.nrows * other.ncols * ctypes.sizeof(ctypes.c_float)
        data = cudart.cuda_malloc(nbytes, ctypes.c_float)
        return cls(data, other.nrows, other.ncols, True)

    @classmethod
    def from_device_array(cls, data, nrows, ncols):
        return cls(data, nrows, ncols, False)

    def to_host(self):
        c_float_p = ctypes.POINTER(ctypes.c_float)
        host_array = (c_float_p * self.nelems)()
        host_ptr = ctypes.cast(host_array, c_float_p)
        elem_size = ctypes.sizeof(ctypes.c_float)
        cudart.cuda_memcpy(host_ptr, self.data, self.nelems * elem_size, 'device_to_host')
        return np.ndarray(shape=(self.nrows, self.ncols),
                          dtype=np.float32,
                          buffer=host_array,
                          order='F')

    def scale(self, context, alpha, out=None):
        if type(alpha) != ctypes.c_float:
            alpha = ctypes.c_float(alpha)
        if out:
            gpu_matrix_kernels.scale(context.cuda_stream, self.nelems, alpha, self.data, out.data)
        else:
            cublas.cublas_s_scal(context.cublas_handle, self.nelems, alpha, self.data, 1)

    def tanh(self, context, tanh_matrix, derivative_matrix=None):
        if derivative_matrix:
            gpu_matrix_kernels.tanh_der(context.cuda_stream, self.nelems, self.data, tanh_matrix.data, derivative_matrix.data)
        else:
            gpu_matrix_kernels.tanh(context.cuda_stream, self.nelems, self.data, tanh_matrix.data)

    def sigmoid(self, context, sigmoid_matrix, derivative_matrix=None):
        if derivative_matrix:
            gpu_matrix_kernels.sigmoid_der(context.cuda_stream, self.nelems, self.data, sigmoid_matrix.data, derivative_matrix.data)
        else:
            gpu_matrix_kernels.sigmoid(context.cuda_stream, self.nelems, self.data, sigmoid_matrix.data)

    def add_scaled(self, context, alpha, a):
        """
        self += alpha * a
        """
        if type(alpha) != ctypes.c_float:
            alpha = ctypes.c_float(alpha)
        cublas.cublas_s_axpy(context.cublas_handle, self.nelems, alpha, a.data, 1, self.data, 1)

    def add(self, context, a, b=None, c=None):
        if not b and not c:
            self.add_scaled(context, 1.0, a)
        else:
            gpu_matrix_kernels.sum(context.cuda_stream, self.nelems, a.data, b.data, c.data, self.data, self.data)

    def sliced_add(self, context, a, column_indxs, alpha=ctypes.c_float(1.0)):
        """
        self[column_indxs] += alpha * a
        """
        if type(alpha) != ctypes.c_float:
            alpha = ctypes.c_float(alpha)
        gpu_matrix_kernels.sliced_inplace_add(context.cuda_stream, a.nrows, a.ncols, alpha, a.data, column_indxs, self.data)

    def add_hprod(self, context, a, b, alpha=ctypes.c_float(1.0)):
        """
        self = a .* b + alpha * self
        """
        if type(alpha) != ctypes.c_float:
            alpha = ctypes.c_float(alpha)
        gpu_matrix_kernels.add_hadamard_product(context.cuda_stream, self.nelems, a.data, b.data, alpha, self.data)

    @staticmethod
    def hprod(context, out, a, b, c=None):
        """
        out = a .* b .* c  or
        out = a .* b
        """
        if not c:
            gpu_matrix_kernels.hadamard_product_2(context.cuda_stream, a.nelems, a.data, b.data, out.data)
        else:
            gpu_matrix_kernels.hadamard_product_3(context.cuda_stream, a.nelems, a.data, b.data, c.data, out.data)

    @staticmethod
    def sum_hprod(context, out, a, b, c, d, e=None, f=None, g=None, h=None, i=None, j=None, k=None):
        """
        out = a .* b + c .* d                                   or
        out = a .* b .* c + d .* e                              or
        out = a .* b .* c + d .* e + f .* g + h .* i + j .* k
        """
        if k is not None:
            gpu_matrix_kernels.sum_hprod_11(context.cuda_stream, out.nelems, a.data, b.data, c.data, d.data, e.data, f.data, g.data, h.data, i.data, j.data, k.data, out.data)
        elif e is not None:
            gpu_matrix_kernels.sum_hprod_5(context.cuda_stream, out.nelems, a.data, b.data, c.data, d.data, e.data, out.data)
        else:
            gpu_matrix_kernels.sum_hprod_4(context.cuda_stream, out.nelems, a.data, b.data, c.data, d.data, out.data)

    def add_dot(self, context, a, b, matrix_operation='N', alpha=ctypes.c_float(1.0), beta=ctypes.c_float(1.0)):
        """
        self = alpha * op(a) * b + beta * self
        """
        if type(alpha) != ctypes.c_float:
            alpha = ctypes.c_float(alpha)
        if type(beta) != ctypes.c_float:
            beta = ctypes.c_float(beta)

        if self.ncols == 1:
            cublas.cublas_s_gemv(context.cublas_handle, matrix_operation, a.nrows, a.ncols, alpha, a.data, a.nrows, b.data, 1, beta, self.data, 1)
        else:
            cublas.cublas_s_gemm(context.cublas_handle, matrix_operation, 'N', self.nrows, self.ncols, b.nrows, alpha, a.data, a.nrows, b.data, b.nrows, beta, self.data, self.nrows)

    def vdot(self, context, a):
        result = ctypes.c_float()
        cublas.cublas_s_dot(context.cublas_handle, self.nelems, self.data, 1, a.data, 1, result)
        return result