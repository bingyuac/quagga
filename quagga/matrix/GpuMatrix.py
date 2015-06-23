import atexit
import numpy as np
import ctypes as ct
from quagga.cuda import cudart, cublas, gpu_matrix_kernels, nonlinearities


class GpuMatrix(object):
    zero_scalar = None
    one_scalar = None
    minus_one_scalar = None
    minus_two_scalar = None

    def __init__(self, data, nrows, ncols, dtype, device_id, is_owner):
        self.data = data
        self.nrows = nrows
        self.ncols = ncols
        self.dtype = dtype
        self.np_dtype, self.c_dtype = self.str_to_dtypes(dtype)
        self.device_id = device_id
        self.is_owner = is_owner
        if is_owner:
            atexit.register(cudart.cuda_free, self.data)

    @property
    def nelems(self):
        return self.nrows * self.ncols

    @property
    def nbytes(self):
        return self.nelems * ct.sizeof(self.c_dtype)

    def __del__(self):
        if self.is_owner:
            try:
                atexit._exithandlers.remove((cudart.cuda_free, (self.data, ), {}))
                cudart.cuda_free(self.data)
            except ValueError:
                pass

    def __getitem__(self, key):
        if type(key[1]) == int:
            data = self._get_pointer_to_column(key[1])
            return GpuMatrix(data, self.nrows, 1, self.dtype, self.device_id, False)
        if type(key[1]) == slice:
            if key[1].start is None and type(key[1].stop) == int and key[1].step is None:
                return GpuMatrix(self.data, self.nrows, key[1].stop, self.dtype, self.device_id, False)
            elif type(key[1].start) == int and key[1].stop is None and key[1].step is None:
                data = self._get_pointer_to_column(key[1].start)
                return GpuMatrix(data, self.nrows, self.ncols - key[1].start, self.dtype, self.device_id, False)
            elif type(key[1].start) == int and type(key[1].stop) == int and key[1].step is None:
                data = self._get_pointer_to_column(key[1].start)
                return GpuMatrix(data, self.nrows, key[1].stop - key[1].start, self.dtype, self.device_id, False)
            else:
                raise ValueError('This slice: {} is unsupported!'.format(key))
        else:
            raise IndexError('Only integers and slices are supported!')

    def same_shape(self, other):
        return self.nrows == other.nrows and self.ncols == other.ncols

    def _get_pointer_to_column(self, k):
        void_p = ct.cast(self.data, ct.c_void_p).value + self.nrows * k * ct.sizeof(self.c_dtype)
        return ct.cast(void_p, ct.POINTER(self.c_dtype))

    @staticmethod
    def str_to_dtypes(dtype):
        if dtype == 'float':
            return np.float32, ct.c_float
        if dtype == 'int':
            return np.int32, ct.c_int
        raise TypeError(u'data type {} not understood'.format(dtype))

    @staticmethod
    def array_to_dtypes(a):
        if a.dtype == np.float32:
            return 'float', np.float32, ct.c_float
        if a.dtype == np.int32:
            return 'int', np.int32, ct.c_int
        raise TypeError(u'data type {} not understood'.format(a.dtype))

    @classmethod
    def from_npa(cls, a, dtype=None, device_id=None):
        if a.ndim != 2:
            raise ValueError('GpuMatrix works only with 2-d numpy arrays!')
        if dtype:
            np_dtype, c_dtype = cls.str_to_dtypes(dtype)
        else:
            dtype, np_dtype, c_dtype = cls.array_to_dtypes(a)
        if not np.isfortran(a):
            a = np.asfortranarray(a, dtype=np_dtype)
        elif a.dtype != np_dtype:
            a = a.astype(dtype=np_dtype)
        host_data = a.ctypes.data_as(ct.POINTER(c_dtype))
        elem_size = ct.sizeof(c_dtype)
        nbytes = a.size * elem_size
        with cudart.device(device_id):
            device_id = cudart.cuda_get_device()
            data = cudart.cuda_malloc(nbytes, c_dtype)
            cudart.cuda_memcpy(data, host_data, nbytes, 'host_to_device')
        return cls(data, a.shape[0], a.shape[1], dtype, device_id, True)

    @classmethod
    def empty(cls, nrows, ncols, dtype, device_id=None):
        c_dtype = cls.str_to_dtypes(dtype)[1]
        nbytes = nrows * ncols * ct.sizeof(c_dtype)
        with cudart.device(device_id):
            device_id = cudart.cuda_get_device()
            data = cudart.cuda_malloc(nbytes, c_dtype)
        return cls(data, nrows, ncols, dtype, device_id, True)

    @classmethod
    def empty_like(cls, other, device_id=None):
        nbytes = other.nelems * ct.sizeof(other.c_dtype)
        with cudart.device(device_id):
            device_id = cudart.cuda_get_device()
            data = cudart.cuda_malloc(nbytes, other.c_dtype)
        return cls(data, other.nrows, other.ncols, other.dtype, device_id, True)

    def to_device(self, context, a, nrows=None, ncols=None):
        """
        This method transfer data from `a` to allocated gpu memory

        :param context: context in which transfer will occur
        :param a: numpy array or ctypes pointer
        :param nrows: optional, is used when `a` is a pointer
        :param ncols: optional, is used when `a` is a pointer
        """

        if type(a) is np.ndarray:
            if self.np_dtype != a.dtype:
                raise ValueError("Allocated memory has {} type. "
                                 "Can't transfer to the device {} type".
                                 format(self.np_dtype, a.dtype))
            if a.ndim != 2:
                raise ValueError('GpuMatrix works only with 2-d numpy arrays!')
            if not np.isfortran(a):
                a = np.asfortranarray(a)
            self.nrows, self.ncols = a.shape
            a = a.ctypes.data_as(ct.POINTER(self.c_dtype))
        else:
            if a._type_ != self.dtype:
                raise ValueError("Allocated memory has {} type. "
                                 "Can't transfer to the device {} type".
                                 format(self.dtype, a._type_))
            self.nrows, self.ncols = nrows, ncols
        context.activate()
        cudart.cuda_memcpy_async(self.data, a, self.nbytes, 'host_to_device', context.cuda_stream)

    def to_host(self):
        c_dtype_p = ct.POINTER(self.c_dtype)
        host_array = (self.c_dtype * self.nelems)()
        host_ptr = ct.cast(host_array, c_dtype_p)
        with cudart.device(self.device_id):
            cudart.cuda_memcpy(host_ptr, self.data, self.nbytes, 'device_to_host')
        return np.ndarray(shape=(self.nrows, self.ncols),
                          dtype=self.np_dtype,
                          buffer=host_array,
                          order='F')

    def to_list(self):
        return [self[:, i] for i in xrange(self.ncols)]

    def copy(self, context, out):
        context.activate()
        cudart.cuda_memcpy_async(out.data, self.data, self.nbytes, 'device_to_device', context.cuda_stream)

    def ravel(self):
        return GpuMatrix(self.data, self.nelems, 1, self.dtype, self.device_id, False)

    def reshape(self, nrows, ncols):
        return GpuMatrix(self.data, nrows, ncols, self.dtype, self.device_id, False)

    def negate(self, context):
        context.activate()
        cublas.cublas_s_axpy(context.cublas_handle, self.nelems, GpuMatrix.minus_two_scalar[context.device_id].data, self.data, 1, self.data, 1)

    def slice_columns(self, context, column_indxs, out):
        if any(context.device_id != device_id for device_id in [self.device_id, column_indxs.device_id, out.device_id]):
            raise ValueError('Matrices have to be on the same device as context!')
        context.activate()
        gpu_matrix_kernels.slice_columns(context.cuda_stream, out.nrows, out.ncols, column_indxs.data, self.data, out.data)

    def assign_hstack(self, context, *matrices):
        # if f_matrix.nrows != s_matrix.nrows:
        #     raise ValueError("Can't horizontally stack matrices with "
        #                      "different number of rows!")
        context.activate()

    def assign_vstack(self, context, *matrices):
        # if self.f_matrix.ncols != self.s_matrix.ncols:
        #     raise ValueError("Can't vertically stack matrices with "
        #                      "different number of columns!")
        context.activate()

    def scale(self, context, alpha, out=None):
        context.activate()
        if out:
            gpu_matrix_kernels.scale(context.cuda_stream, self.nelems, alpha.data, self.data, out.data)
        else:
            cublas.cublas_s_scal(context.cublas_handle, self.nelems, alpha.data, self.data, 1)

    def tanh(self, context, tanh_matrix, derivative_matrix=None):
        context.activate()
        if derivative_matrix:
            nonlinearities.tanh_der(context.cuda_stream, self.nelems, self.data, tanh_matrix.data, derivative_matrix.data)
        else:
            nonlinearities.tanh(context.cuda_stream, self.nelems, self.data, tanh_matrix.data)

    def sigmoid(self, context, sigmoid_matrix, derivative_matrix=None):
        context.activate()
        if derivative_matrix:
            nonlinearities.sigmoid_der(context.cuda_stream, self.nelems, self.data, sigmoid_matrix.data, derivative_matrix.data)
        else:
            nonlinearities.sigmoid(context.cuda_stream, self.nelems, self.data, sigmoid_matrix.data)

    def tanh_sigm(self, context, tanh_sigm_matrix, derivative_matrix=None):
        """
        This is a fancy function that is used during forward propagation into
        lstm cell. It calculates for the first 1/4 rows tanh function and
        sigmoid for the 3/4 remaining rows.
        """
        context.activate()
        if derivative_matrix:
            nonlinearities.tanh_sigm_der(context.cuda_stream, self.nrows, self.ncols, self.data, tanh_sigm_matrix.data, derivative_matrix.data)
        else:
            nonlinearities.tanh_sigm(context.cuda_stream, self.nrows, self.ncols, self.data, tanh_sigm_matrix.data)

    def softmax(self, context, softmax_matrix):
        # TODO
        context.activate()

    def add_scaled(self, context, alpha, a):
        """
        self += alpha * a
        """
        context.activate()
        cublas.cublas_s_axpy(context.cublas_handle, self.nelems, alpha.data, a.data, 1, self.data, 1)

    def add(self, context, a, b=None, c=None):
        if not b and not c:
            self.add_scaled(context, GpuMatrix.one_scalar[context.device_id], a)
        else:
            context.activate()
            gpu_matrix_kernels.sum(context.cuda_stream, self.nelems, a.data, b.data, c.data, self.data, self.data)

    def sub(self, context, a):
        self.add_scaled(context, GpuMatrix.minus_one_scalar[context.device_id], a)

    def sliced_add_scaled(self, context, column_indxs, alpha, a):
        """
        self[column_indxs] += alpha * a
        """
        context.activate()
        gpu_matrix_kernels.sliced_inplace_add(context.cuda_stream, a.nrows, a.ncols, alpha.data, a.data, column_indxs.data, self.data)

    def sliced_add(self, context, column_indxs, a):
        """
        self[column_indxs] += a
        """
        self.sliced_add_scaled(context, column_indxs, GpuMatrix.one_scalar[context.device_id], a)

    def add_hprod(self, context, a, b, alpha=None):
        """
        self = a .* b + alpha * self
        """
        alpha = alpha if alpha else GpuMatrix.one_scalar[context.device_id]
        context.activate()
        gpu_matrix_kernels.add_hadamard_product(context.cuda_stream, self.nelems, a.data, b.data, alpha.data, self.data)

    def assign_hprod(self, context, a, b, c=None):
        """
        self = a .* b
        self = a .* b .* c  or
        """
        context.activate()
        if not c:
            gpu_matrix_kernels.hadamard_product_2(context.cuda_stream, a.nelems, a.data, b.data, self.data)
        else:
            gpu_matrix_kernels.hadamard_product_3(context.cuda_stream, a.nelems, a.data, b.data, c.data, self.data)

    def assign_sum_hprod(self, context, a, b, c, d, e=None, f=None, g=None, h=None, i=None, j=None, k=None):
        """
        self = a .* b + c .* d                                   or
        self = a .* b .* c + d .* e                              or
        self = a .* b .* c + d .* e + f .* g + h .* i + j .* k
        """
        context.activate()
        if k is not None:
            gpu_matrix_kernels.sum_hprod_11(context.cuda_stream, self.nelems, a.data, b.data, c.data, d.data, e.data, f.data, g.data, h.data, i.data, j.data, k.data, self.data)
        elif e is not None:
            gpu_matrix_kernels.sum_hprod_5(context.cuda_stream, self.nelems, a.data, b.data, c.data, d.data, e.data, self.data)
        else:
            gpu_matrix_kernels.sum_hprod_4(context.cuda_stream, self.nelems, a.data, b.data, c.data, d.data, self.data)

    def assign_hprod_sum(self, context, a, b):
        """
        self = sum(a .* b, axis=1)
        """
        context.activate()
        gpu_matrix_kernels.hprod_sum(context.cuda_stream, a.nrows, a.ncols, a.data, b.data, self.data)

    def assign_dot(self, context, a, b, matrix_operation_a='N', matrix_operation_b='N'):
        self.add_dot(context, a, b, matrix_operation_a, matrix_operation_b, beta=GpuMatrix.zero_scalar[context.device_id])

    def add_dot(self, context, a, b, matrix_operation_a='N', matrix_operation_b='N', alpha=None, beta=None):
        """
        self = alpha * op(a) * b + beta * self
        """
        alpha = alpha if alpha else GpuMatrix.one_scalar[context.device_id]
        beta = beta if beta else GpuMatrix.one_scalar[context.device_id]
        context.activate()
        if self.ncols == 1 and matrix_operation_b == 'N':
            cublas.cublas_s_gemv(context.cublas_handle, matrix_operation_a, a.nrows, a.ncols, alpha.data, a.data, a.nrows, b.data, 1, beta.data, self.data, 1)
        else:
            k = b.nrows if matrix_operation_b == 'N' else b.ncols
            cublas.cublas_s_gemm(context.cublas_handle, matrix_operation_a, matrix_operation_b, self.nrows, self.ncols, k, alpha.data, a.data, a.nrows, b.data, b.nrows, beta.data, self.data, self.nrows)

    def assign_vdot(self, context, a):
        context.activate()
        cublas.cublas_s_dot(context.cublas_handle, self.nelems, self.data, 1, a.data, 1, self.data)


GpuMatrix.zero_scalar = []
GpuMatrix.one_scalar = []
GpuMatrix.minus_one_scalar = []
GpuMatrix.minus_two_scalar = []
for device_id in xrange(cudart.cuda_get_device_count()):
    GpuMatrix.zero_scalar.append(GpuMatrix.from_npa(np.zeros((1, 1)), 'float', device_id))
    GpuMatrix.one_scalar.append(GpuMatrix.from_npa(np.ones((1, 1)), 'float', device_id))
    GpuMatrix.minus_one_scalar.append(GpuMatrix.from_npa(-np.ones((1, 1)), 'float', device_id))
    GpuMatrix.minus_two_scalar.append(GpuMatrix.from_npa(-2*np.ones((1, 1)), 'float', device_id))