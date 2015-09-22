import quagga
import weakref
import warnings
import numpy as np
import ctypes as ct
from itertools import chain
from quagga.cuda import cudnn
from quagga.cuda import cudart
from quagga.cuda import cublas
from quagga.cuda import curand
from quagga.cuda import nonlinearities
from quagga.matrix import ShapeElement
from quagga.cuda import gpu_matrix_kernels


warning_messages = ['P2P comunication is not possible between:']
for i in xrange(cudart.cuda_get_device_count()):
    for j in xrange(cudart.cuda_get_device_count()):
        if i != j:
            if not cudart.cuda_device_can_access_peer(i, j):
                warning_messages.append('GPU{}->GPU{}'.format(i, j))
if len(warning_messages) != 1:
    warnings.warn(' '.join(warning_messages), UserWarning)


class GpuMatrix(object):
    def __init__(self, data, nrows, ncols, dtype, device_id, is_owner, strides=None, base=None):
        self.data = data
        self._nrows = nrows if isinstance(nrows, ShapeElement) else ShapeElement(nrows)
        self._ncols = ncols if isinstance(ncols, ShapeElement) else ShapeElement(ncols)
        self.dtype = dtype
        self.np_dtype, self.c_dtype = self.str_to_dtypes(dtype)
        self._cudnn_tensor_descriptor = None
        self.device_id = device_id
        self.is_owner = is_owner
        weak_self = weakref.ref(self)
        if strides:
            self.strides = strides
        else:
            elem_size = ct.sizeof(self.c_dtype)
            self.strides = [elem_size, self.nrows * elem_size]
            change_strides = lambda: weak_self().strides.__setitem__(1, weak_self().nrows * elem_size)
            self.nrows.add_modification_handler(change_strides)
        self.base = base
        self.last_modification_context = None

        def change_cudnn_tensor_descriptor():
            strong_self = weak_self()
            if strong_self._cudnn_tensor_descriptor:
                cudnn.destroy_tensor_descriptor(strong_self._cudnn_tensor_descriptor)
                strong_self._cudnn_tensor_descriptor = None
        self.nrows.add_modification_handler(change_cudnn_tensor_descriptor)
        self.ncols.add_modification_handler(change_cudnn_tensor_descriptor)

    @staticmethod
    def get_setable_attributes():
        return ['nrows', 'ncols', 'last_modification_context']

    @property
    def nelems(self):
        return self._nrows.value * self._ncols.value

    @property
    def nrows(self):
        return self._nrows

    @nrows.setter
    def nrows(self, value):
        self._nrows[:] = value

    @property
    def ncols(self):
        return self._ncols

    @ncols.setter
    def ncols(self, value):
        self._ncols[:] = value

    @property
    def nbytes(self):
        return self.nelems * ct.sizeof(self.c_dtype)

    @property
    def cudnn_tensor_descriptor(self):
        if not self._cudnn_tensor_descriptor:
            self._cudnn_tensor_descriptor = cudnn.ct_cudnn_tensor_descriptor()
            cudnn.create_tensor_descriptor(self._cudnn_tensor_descriptor)
            # CUDNN uses C-order, but CUBLAS uses F-order
            cudnn.set_tensor_4d_descriptor_ex(self._cudnn_tensor_descriptor,
                                              cudnn.data_type['CUDNN_DATA_FLOAT'],
                                              self.nrows, self.ncols, 1, 1,
                                              1, self.nrows, 1, 1)
        return self._cudnn_tensor_descriptor

    def __del__(self):
        if self.is_owner:
            cudart.cuda_free(self.data)
            if self._cudnn_tensor_descriptor:
                cudnn.destroy_tensor_descriptor(self._cudnn_tensor_descriptor)

    def __getitem__(self, key):
        # get row
        weak_self = weakref.ref(self)
        if isinstance(key, int):
            data = self._get_pointer_to_element(key, 0)
            return GpuMatrix(data, 1, self.ncols, self.dtype, self.device_id, False, self.strides, self)
        if isinstance(key, ShapeElement):
            data = self._get_pointer_to_element(key.value, 0)
            a = GpuMatrix(data, 1, self.ncols, self.dtype, self.device_id, False, self.strides, self)
            modif_handler = lambda: setattr(a, 'data', weak_self()._get_pointer_to_element(key.value, 0))
            key.add_modification_handler(modif_handler)
            return a
        if isinstance(key, slice) and self.ncols == 1:
            key = (key, 0)
        # get row slice with one column
        if isinstance(key[0], slice) and not key[0].step and isinstance(key[1], (int, ShapeElement)):
            start = key[0].start if key[0].start else 0
            stop = key[0].stop if key[0].stop else self.nrows
            nrows = stop - start
            if isinstance(start, int) and isinstance(key[1], int):
                data = self._get_pointer_to_element(start, key[1])
                return GpuMatrix(data, nrows, 1, self.dtype, self.device_id, False, self.strides, self)
            elif isinstance(start, int) and isinstance(key[1], ShapeElement):
                data = self._get_pointer_to_element(start, key[1].value)
                a = GpuMatrix(data, nrows, 1, self.dtype, self.device_id, False, self.strides, self)
                modif_handler = lambda: setattr(a, 'data', weak_self()._get_pointer_to_element(start, key[1].value))
                key[1].add_modification_handler(modif_handler)
                return a
            elif isinstance(start, ShapeElement) and isinstance(key[1], int):
                data = self._get_pointer_to_element(start.value, key[1])
                a = GpuMatrix(data, nrows, 1, self.dtype, self.device_id, False, self.strides, self)
                modif_handler = lambda: setattr(a, 'data', weak_self()._get_pointer_to_element(start.value, key[1]))
                start.add_modification_handler(modif_handler)
                return a
            elif isinstance(start, ShapeElement) and isinstance(key[1], ShapeElement):
                data = self._get_pointer_to_element(start.value, key[1].value)
                a = GpuMatrix(data, nrows, 1, self.dtype, self.device_id, False, self.strides, self)
                modif_handler = lambda: setattr(a, 'data', weak_self()._get_pointer_to_element(start.value, key[1].value))
                key[1].add_modification_handler(modif_handler)
                start.add_modification_handler(modif_handler)
                return a
        # get column slice
        if key[0] == slice(None) and isinstance(key[1], slice) and not key[1].step:
            stop = key[1].stop if key[1].stop else self.ncols
            start = key[1].start if key[1].start else 0
            ncols = stop - start
            if isinstance(start, int):
                data = self._get_pointer_to_column(start)
                return GpuMatrix(data, self.nrows, ncols, self.dtype, self.device_id, False, base=self)
            elif isinstance(start, ShapeElement):
                data = self._get_pointer_to_column(start.value)
                a = GpuMatrix(data, self.nrows, ncols, self.dtype, self.device_id, False, base=self)
                modif_handler = lambda: setattr(a, 'data', weak_self()._get_pointer_to_column(start.value))
                start.add_modification_handler(modif_handler)
                return a
        raise ValueError('This slice: {} is unsupported!'.format(key))

    def same_shape(self, other):
        return self.nrows == other.nrows and self.ncols == other.ncols

    def _get_pointer_to_column(self, k):
        void_p = ct.cast(self.data, ct.c_void_p).value + self.nrows.value * k * ct.sizeof(self.c_dtype)
        return ct.cast(void_p, ct.POINTER(self.c_dtype))

    def _get_pointer_to_row(self, k):
        void_p = ct.cast(self.data, ct.c_void_p).value + k * ct.sizeof(self.c_dtype)
        return ct.cast(void_p, ct.POINTER(self.c_dtype))

    def _get_pointer_to_element(self, i, j):
        void_p = ct.cast(self.data, ct.c_void_p).value + (self.nrows.value * j + i) * ct.sizeof(self.c_dtype)
        return ct.cast(void_p, ct.POINTER(self.c_dtype))

    @staticmethod
    def wait_matrices(current_context, *matrices):
        contexts = set(e.last_modification_context for e in matrices)
        contexts.discard(None)
        contexts.discard(current_context)
        current_context.wait(*contexts)

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
        a_gpu = cls.empty(a.shape[0], a.shape[1], dtype, device_id)
        host_data = a.ctypes.data_as(ct.POINTER(c_dtype))
        cudart.cuda_memcpy(a_gpu.data, host_data, a_gpu.nbytes, 'default')
        return a_gpu

    @classmethod
    def empty(cls, nrows, ncols, dtype=None, device_id=None):
        dtype = dtype if dtype else quagga.dtype
        with cudart.device(device_id):
            device_id = cudart.cuda_get_device()
            a = cls(None, nrows, ncols, dtype, device_id, True)
            a.data = cudart.cuda_malloc(a.nbytes, a.c_dtype)
        return a

    @classmethod
    def empty_like(cls, other, device_id=None):
        device_id = other.device_id if device_id is None else device_id
        return cls.empty(other.nrows, other.ncols, other.dtype, device_id)

    def to_host(self, context=None):
        if context:
            GpuMatrix.wait_matrices(context, self)
            context.activate()
        c_dtype_p = ct.POINTER(self.c_dtype)
        host_array = (self.c_dtype * self.nelems)()
        host_ptr = ct.cast(host_array, c_dtype_p)
        with cudart.device(self.device_id):
            if self.nrows == 1 and self.strides[0] != self.strides[1]:
                dpitch = ct.c_size_t(self.strides[0])
                spitch = ct.c_size_t(self.strides[1])
                width = ct.c_size_t(self.strides[0])
                height = ct.c_size_t(self.nelems)
                args = [host_ptr, dpitch, self.data, spitch, width, height, 'default']
                if context:
                    fun = cudart.cuda_memcpy2d_async
                    args.append(context.cuda_stream)
                else:
                    fun = cudart.cuda_memcpy2d
            else:
                args = [host_ptr, self.data, self.nbytes, 'default']
                if context:
                    fun = cudart.cuda_memcpy_async
                    args.append(context.cuda_stream)
                else:
                    fun = cudart.cuda_memcpy
            fun(*args)
        return np.ndarray(shape=(self.nrows, self.ncols),
                          dtype=self.np_dtype,
                          buffer=host_array,
                          order='F')

    def assign(self, context, a):
        """
        self <- a
        """

        GpuMatrix.wait_matrices(context, a)
        self.last_modification_context = context
        context.activate()

        # TODO(sergii): add real stride support
        if a.nrows == 1 and self.nrows == 1 and (a.strides[0] != a.strides[1] or self.strides[0] != self.strides[1]):
            dpitch = ct.c_size_t(self.strides[1])
            spitch = ct.c_size_t(a.strides[1])
            width = ct.c_size_t(a.strides[0])
            height = ct.c_size_t(a.nelems)
            cudart.cuda_memcpy2d_async(self.data, dpitch, a.data, spitch, width, height, 'default', context.cuda_stream)
        else:
            cudart.cuda_memcpy_async(self.data, a.data, a.nbytes, 'default', context.cuda_stream)

    def assign_npa(self, context, a, nrows=None, ncols=None):
        """
        This method transfer data from `a` to allocated gpu memory

        :param context: context in which transfer will occur
        :param a: numpy array or ctypes pointer
        :param nrows: optional, is used when `a` is a pointer
        :param ncols: optional, is used when `a` is a pointer
        """

        self.last_modification_context = context
        context.activate()

        if isinstance(a, np.ndarray):
            if self.np_dtype != a.dtype:
                raise ValueError("Allocated memory has {} type. "
                                 "Can't transfer {} type".
                                 format(self.np_dtype, a.dtype))
            if a.ndim != 2:
                raise ValueError('GpuMatrix works only with 2-d numpy arrays!')
            if not np.isfortran(a):
                a = np.asfortranarray(a)
            self.nrows, self.ncols = a.shape
            host_data = a.ctypes.data_as(ct.POINTER(self.c_dtype))
        else:
            # this branch for ctypes array
            if a._type_ != self.dtype:
                raise ValueError("Allocated memory has {} type. "
                                 "Can't transfer {} type".
                                 format(self.dtype, a._type_))
            self.nrows, self.ncols = nrows, ncols
            host_data = a
        cudart.cuda_memcpy_async(self.data, host_data, self.nbytes, 'default', context.cuda_stream)

    def fill(self, context, value):
        self.last_modification_context = context
        context.activate()
        gpu_matrix_kernels.fill(context.cuda_stream, self.nelems, value, self.data)

    def sync_fill(self, value):
        a = np.empty((self.nrows, self.ncols), self.np_dtype, 'F')
        a.fill(value)
        host_data = a.ctypes.data_as(ct.POINTER(self.c_dtype))
        with cudart.device(self.device_id):
            cudart.cuda_memcpy(self.data, host_data, self.nbytes, 'default')

    def slice_columns(self, context, column_indxs, out, reverse=False):
        GpuMatrix.wait_matrices(context, self, column_indxs)
        out.last_modification_context = context
        context.activate()
        if reverse:
            gpu_matrix_kernels.reverse_slice_columns(context.cuda_stream, out.nrows, out.ncols, column_indxs.data, self.data, out.data)
        else:
            gpu_matrix_kernels.slice_columns(context.cuda_stream, out.nrows, out.ncols, column_indxs.data, self.data, out.data)

    def add_scaled_columns_slice(self, context, column_indxs, alpha, a):
        """
        self[column_indxs] += alpha * a
        """
        GpuMatrix.wait_matrices(context, self, column_indxs, a)
        self.last_modification_context = context
        context.activate()
        gpu_matrix_kernels.sliced_inplace_add(context.cuda_stream, a.nrows, a.ncols, alpha, a.data, column_indxs.data, self.data)

    def add_columns_slice(self, context, column_indxs, a):
        """
        self[column_indxs] += a
        """
        self.add_scaled_columns_slice(context, column_indxs, 1.0, a)

    def slice_columns_and_transpose(self, context, column_indxs, out):
        GpuMatrix.wait_matrices(context, self, column_indxs)
        out.last_modification_context = context
        context.activate()
        gpu_matrix_kernels.slice_columns_and_transpose(context.cuda_stream, out.nrows, out.ncols, column_indxs.data, self.data, out.data)

    def slice_rows(self, context, row_indxs, out):
        GpuMatrix.wait_matrices(context, self, row_indxs)
        out.last_modification_context = context
        context.activate()
        gpu_matrix_kernels.slice_rows(context.cuda_stream, self.nrows, row_indxs.data, self.data, out.nrows, out.ncols, out.data)

    def slice_rows_batch(self, context, embd_rows_indxs, dense_matrices):
        GpuMatrix.wait_matrices(context, self, embd_rows_indxs)
        for dense_matrix in dense_matrices:
            dense_matrix.last_modification_context = context
        context.activate()

        n = len(dense_matrices)
        matrices = (ct.POINTER(self.c_dtype) * n)(*(m.data for m in dense_matrices))
        device_pointer = _get_temp_memory(context, n)
        elem_size = ct.sizeof(ct.POINTER(ct.c_float))
        cudart.cuda_memcpy_async(device_pointer, matrices, n * elem_size, 'default', context.cuda_stream)
        gpu_matrix_kernels.slice_rows_batch(context.cuda_stream, embd_rows_indxs.data, embd_rows_indxs.nrows, embd_rows_indxs.ncols, self.data, self.nrows, self.ncols, device_pointer)

    def add_scaled_rows_batch_slice(self, context, embd_rows_indxs, alpha, dense_matrices):
        """
        for k in range(K):
            self[embd_rows_indxs[:, k]] += alpha * dense_matrices[k]
        """

        GpuMatrix.wait_matrices(context, embd_rows_indxs, *dense_matrices)
        self.last_modification_context = context
        context.activate()

        n = len(dense_matrices)
        matrices = (ct.POINTER(self.c_dtype) * n)(*(m.data for m in dense_matrices))
        device_pointer = _get_temp_memory(context, n)
        elem_size = ct.sizeof(ct.POINTER(ct.c_float))
        cudart.cuda_memcpy_async(device_pointer, matrices, n * elem_size, 'default', context.cuda_stream)
        gpu_matrix_kernels.sliced_rows_batch_scaled_add(context.cuda_stream, embd_rows_indxs.data, embd_rows_indxs.nrows, embd_rows_indxs.ncols, alpha, device_pointer, self.nrows, self.ncols, self.data)

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
        self.wait_matrices(context, *matrices)
        self.last_modification_context = context
        context.activate()

        n = len(matrices)
        ncols = (ct.c_int * n)(*(m.ncols for m in matrices))
        matrices = (ct.POINTER(self.c_dtype) * n)(*(m.data for m in matrices))
        gpu_matrix_kernels.horizontal_stack(context.cuda_stream, n, ncols, self.nrows, matrices, self.data)

    def hsplit(self, context, matrices, col_slices=None):
        GpuMatrix.wait_matrices(context, self)
        for matrix in matrices:
            matrix.last_modification_context = context
        context.activate()

        n = len(matrices)
        if col_slices:
            max_col = -np.inf
            for col_slice in col_slices:
                max_col = col_slice[1] if col_slice[1] > max_col else max_col
            if max_col > self.ncols:
                raise ValueError("One of the slice does not match with the array size!")
            col_slices = (ct.c_int * (2 * n))(*(sum(col_slices, ())))
            matrices = (ct.POINTER(self.c_dtype) * n)(*(m.data for m in matrices))
            gpu_matrix_kernels.horizontal_slice_split(context.cuda_stream, n, col_slices, self.nrows, matrices, self.data)
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
            ncols = (ct.c_int * n)(*(m.ncols for m in matrices))
            matrices = (ct.POINTER(self.c_dtype) * n)(*(m.data for m in matrices))
            gpu_matrix_kernels.hotizontal_split(context.cuda_stream, n, ncols, self.nrows, matrices, self.data)

    @staticmethod
    def batch_hstack(context, x_sequence, y_sequence, output_sequence):
        GpuMatrix.wait_matrices(context, *chain(x_sequence, y_sequence))
        for matrix in output_sequence:
            matrix.last_modification_context = context
        context.activate()

        n = len(output_sequence)
        c_dtype = output_sequence[0].c_dtype
        x_matrices = (ct.POINTER(c_dtype) * n)(*(m.data for m in x_sequence))
        y_matrices = (ct.POINTER(c_dtype) * n)(*(m.data for m in y_sequence))
        output_matrices = (ct.POINTER(c_dtype) * n)(*(m.data for m in output_sequence))
        elem_size = ct.sizeof(ct.POINTER(c_dtype))
        x_device_pointer = _get_temp_memory(context, 3 * n)
        cudart.cuda_memcpy_async(x_device_pointer, x_matrices, n * elem_size, 'default', context.cuda_stream)

        void_p = ct.cast(x_device_pointer, ct.c_void_p).value + n * elem_size
        y_device_pointer = ct.cast(void_p, ct.POINTER(ct.POINTER(c_dtype)))
        cudart.cuda_memcpy_async(y_device_pointer, y_matrices, n * elem_size, 'default', context.cuda_stream)

        void_p = ct.cast(y_device_pointer, ct.c_void_p).value + n * elem_size
        output_device_pointer = ct.cast(void_p, ct.POINTER(ct.POINTER(c_dtype)))
        cudart.cuda_memcpy_async(output_device_pointer, output_matrices, n * elem_size, 'default', context.cuda_stream)

        nrows = output_sequence[0].nrows
        x_ncols = x_sequence[0].ncols
        y_ncols = y_sequence[0].ncols

        gpu_matrix_kernels.batch_horizontal_stack(context.cuda_stream, n, nrows, x_ncols, y_ncols, x_device_pointer, y_device_pointer, output_device_pointer)

    @staticmethod
    def batch_hsplit(context, input_sequence, x_sequence, y_sequence):
        GpuMatrix.wait_matrices(context, *input_sequence)
        for matrix in chain(x_sequence, y_sequence):
            matrix.last_modification_context = context
        context.activate()

        n = len(input_sequence)
        c_dtype = input_sequence[0].c_dtype
        x_matrices = (ct.POINTER(c_dtype) * n)(*(m.data for m in x_sequence))
        y_matrices = (ct.POINTER(c_dtype) * n)(*(m.data for m in y_sequence))
        input_matrices = (ct.POINTER(c_dtype) * n)(*(m.data for m in input_sequence))
        elem_size = ct.sizeof(ct.POINTER(c_dtype))
        x_device_pointer = _get_temp_memory(context, 3 * n)
        cudart.cuda_memcpy_async(x_device_pointer, x_matrices, n * elem_size, 'default', context.cuda_stream)

        void_p = ct.cast(x_device_pointer, ct.c_void_p).value + n * elem_size
        y_device_pointer = ct.cast(void_p, ct.POINTER(ct.POINTER(c_dtype)))
        cudart.cuda_memcpy_async(y_device_pointer, y_matrices, n * elem_size, 'default', context.cuda_stream)

        void_p = ct.cast(y_device_pointer, ct.c_void_p).value + n * elem_size
        input_device_pointer = ct.cast(void_p, ct.POINTER(ct.POINTER(c_dtype)))
        cudart.cuda_memcpy_async(input_device_pointer, input_matrices, n * elem_size, 'default', context.cuda_stream)

        nrows = input_sequence[0].nrows
        x_ncols = x_sequence[0].ncols
        y_ncols = y_sequence[0].ncols

        gpu_matrix_kernels.batch_horizontal_split(context.cuda_stream, n, nrows, x_ncols, y_ncols, input_device_pointer, x_device_pointer, y_device_pointer)

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
        GpuMatrix.wait_matrices(context, *matrices)
        self.last_modification_context = context
        context.activate()

        n = len(matrices)
        nrows = (ct.c_int * n)(*(m.nrows for m in matrices))
        matrices = (ct.POINTER(self.c_dtype) * n)(*(m.data for m in matrices))
        gpu_matrix_kernels.vertical_stack(context.cuda_stream, n, nrows, self.ncols, matrices, self.data)

    def vsplit(self, context, matrices, row_slices=None):
        GpuMatrix.wait_matrices(context, self)
        for m in matrices:
            m.last_modification_context = context
        context.activate()

        n = len(matrices)
        if row_slices:
            max_row = -np.inf
            for row_slice in row_slices:
                max_row = row_slice[1] if row_slice[1] > max_row else max_row
            if max_row > self.nrows:
                raise ValueError("One of the slice does not match with the array size!")
            row_slices = (ct.c_int * (2 * n))(*(sum(row_slices, ())))
            matrices = (ct.POINTER(self.c_dtype) * n)(*(m.data for m in matrices))
            gpu_matrix_kernels.vertical_slice_split(context.cuda_stream, n, row_slices, self.nrows, self.ncols, matrices, self.data)
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
            nrows = (ct.c_int * n)(*(m.nrows for m in matrices))
            matrices = (ct.POINTER(self.c_dtype) * n)(*(m.data for m in matrices))
            gpu_matrix_kernels.vertical_split(context.cuda_stream, n, nrows, self.ncols, matrices, self.data)

    def assign_sequential_mean_pooling(self, context, matrices):
        GpuMatrix.wait_matrices(context, *matrices)
        self.last_modification_context = context
        context.activate()

        n = len(matrices)
        matrices = (ct.POINTER(self.c_dtype) * n)(*(m.data for m in matrices))
        device_pointer = _get_temp_memory(context, n)
        elem_size = ct.sizeof(ct.POINTER(ct.c_float))
        cudart.cuda_memcpy_async(device_pointer, matrices, n * elem_size, 'default', context.cuda_stream)
        self.fill(context, 0.0)
        gpu_matrix_kernels.assign_sequential_mean_pooling(context.cuda_stream, self.nrows, self.ncols, device_pointer, n, self.data)

    @staticmethod
    def sequentially_tile(context, a, matrices):
        for matrix in matrices:
            if matrix.nrows != a.nrows or matrix.ncols != a.ncols:
                raise ValueError('Invalid shape! `a` matrix must have the '
                                 'same number of rows and columns as matrices '
                                 'to be tiled!')
        GpuMatrix.wait_matrices(context, a)
        for matrix in matrices:
            matrix.last_modification_context = context
        context.activate()

        n = len(matrices)
        matrices = (ct.POINTER(a.c_dtype) * n)(*(m.data for m in matrices))
        device_pointer = _get_temp_memory(context, n)
        elem_size = ct.sizeof(ct.POINTER(ct.c_float))
        cudart.cuda_memcpy_async(device_pointer, matrices, n * elem_size, 'default', context.cuda_stream)
        gpu_matrix_kernels.sequentially_tile(context.cuda_stream, a.nelems, a.data, device_pointer, n)

    def tile(self, context, axis, a):
        GpuMatrix.wait_matrices(context, a)
        self.last_modification_context = context
        context.activate()

        if axis == 0:
            if a.nrows != 1:
                raise ValueError('Invalid shape! `a` must have number of rows '
                                 'equal to one!')
            if self.ncols != a.ncols:
                raise ValueError('Invalid shape! `a` matrix must have the '
                                 'same number of columns as matrix to be tiled!')
            for i in xrange(self.nrows):
                row = self._get_pointer_to_row(i)
                cublas.s_copy(context.cublas_handle, self.ncols, a.data, 1, row, self.nrows)
        elif axis == 1:
            if a.ncols != 1:
                raise ValueError('Invalid shape! `a` must have number of '
                                 'columns equal to one!')
            if self.nrows != a.nrows:
                raise ValueError('Invalid shape! `a` matrix must have the '
                                 'same number of rows as matrix to be tiled!')
            for i in xrange(self.ncols):
                column = self._get_pointer_to_column(i)
                cublas.s_copy(context.cublas_handle, self.nrows, a.data, 1, column, 1)
        else:
            raise ValueError('Invalid axis!')

    @staticmethod
    def get_random_generator(seed):
        generator = curand.ct_curand_generator()
        curand.create_generator(generator, curand.curand_rng_type['CURAND_RNG_PSEUDO_DEFAULT'])
        curand.pseudo_random_generator_seed(generator, seed)
        return generator

    def dropout(self, context, generator, dropout_prob, out):
        GpuMatrix.wait_matrices(context, self)
        out.last_modification_context = context
        context.activate()

        curand.set_stream(generator, context.cuda_stream)
        curand.generate_uniform(generator, out.data, self.nelems)
        gpu_matrix_kernels.dropout(context.cuda_stream, self.nelems, dropout_prob, self.data, out.data, out.data)

    def assign_mask_zeros(self, context, a, b):
        """
        self = a * (b != 0)
        """

        GpuMatrix.wait_matrices(context, a, b)
        self.last_modification_context = context
        context.activate()
        gpu_matrix_kernels.mask_zeros(context.cuda_stream, self.nelems, a.data, b.data, self.data)

    def add_mask_zeros(self, context, a, b):
        """
        self += a * (b != 0)
        """

        GpuMatrix.wait_matrices(context, a, b)
        self.last_modification_context = context
        context.activate()
        gpu_matrix_kernels.add_mask_zeros(context.cuda_stream, self.nelems, a.data, b.data, self.data)

    def mask_column_numbers_row_wise(self, context, numbers):
        """
        self[i, j] = j < numbers[i]
        """

        GpuMatrix.wait_matrices(context, numbers)
        self.last_modification_context = context
        context.activate()
        gpu_matrix_kernels.mask_column_numbers_row_wise(context.cuda_stream, self.nrows, self.ncols, numbers.data, self.data)

    def tanh(self, context, tanh_matrix, derivative_matrix=None):
        GpuMatrix.wait_matrices(context, self)
        tanh_matrix.last_modification_context = context
        context.activate()
        if derivative_matrix:
            derivative_matrix.last_modification_context = context
            nonlinearities.tanh_der(context.cuda_stream, self.nelems, self.data, tanh_matrix.data, derivative_matrix.data)
        else:
            nonlinearities.tanh(context.cuda_stream, self.nelems, self.data, tanh_matrix.data)

    def sigmoid(self, context, sigmoid_matrix, derivative_matrix=None):
        GpuMatrix.wait_matrices(context, self)
        sigmoid_matrix.last_modification_context = context
        context.activate()
        if derivative_matrix:
            derivative_matrix.last_modification_context = context
            nonlinearities.sigmoid_der(context.cuda_stream, self.nelems, self.data, sigmoid_matrix.data, derivative_matrix.data)
        else:
            nonlinearities.sigmoid(context.cuda_stream, self.nelems, self.data, sigmoid_matrix.data)

    def tanh_sigm(self, context, tanh_sigm_matrix, derivative_matrix=None, axis=0):
        """
        This is a fancy function that is used during forward propagation into
        lstm cell. It calculates for the first 1/4 elements along the axis
        tanh function and sigmoid for the 3/4 remaining elements.
        """

        GpuMatrix.wait_matrices(context, self)
        tanh_sigm_matrix.last_modification_context = context
        context.activate()

        if axis not in {0, 1}:
            raise ValueError('TODO!')
        if derivative_matrix:
            derivative_matrix.last_modification_context = context
            nonlinearities.tanh_sigm_der(context.cuda_stream, axis, self.nrows, self.ncols, self.data, tanh_sigm_matrix.data, derivative_matrix.data)
        else:
            nonlinearities.tanh_sigm(context.cuda_stream, axis, self.nrows, self.ncols, self.data, tanh_sigm_matrix.data)

    def relu(self, context, relu_matrix, derivative_matrix=None):
        GpuMatrix.wait_matrices(context, self)
        relu_matrix.last_modification_context = context
        context.activate()
        if derivative_matrix:
            derivative_matrix.last_modification_context = context
            nonlinearities.relu_der(context.cuda_stream, self.nelems, self.data, relu_matrix.data, derivative_matrix.data)
        else:
            nonlinearities.relu(context.cuda_stream, self.nelems, self.data, relu_matrix.data)

    def softmax(self, context, softmax_matrix):
        GpuMatrix.wait_matrices(context, self)
        softmax_matrix.last_modification_context = context
        context.activate()
        cudnn.softmax_forward(context.cudnn_handle,
                              cudnn.softmax_algorithm['CUDNN_SOFTMAX_ACCURATE'],
                              cudnn.softmax_mode['CUDNN_SOFTMAX_MODE_INSTANCE'],
                              ct.c_float(1.0),
                              self.cudnn_tensor_descriptor,
                              self.data,
                              ct.c_float(0.0),
                              softmax_matrix.cudnn_tensor_descriptor,
                              softmax_matrix.data)

    def assign_softmax_ce_derivative(self, context, probs, target_classes):
        GpuMatrix.wait_matrices(context, probs, target_classes)
        self.last_modification_context = context
        context.activate()
        gpu_matrix_kernels.softmax_ce_derivative(context.cuda_stream, probs.nrows, probs.ncols, probs.data, target_classes.data, self.data)

    def add_softmax_ce_derivative(self, context, probs, target_classes):
        GpuMatrix.wait_matrices(context, probs, target_classes)
        self.last_modification_context = context
        context.activate()
        gpu_matrix_kernels.add_softmax_ce_derivative(context.cuda_stream, probs.nrows, probs.ncols, probs.data, target_classes.data, self.data)

    def scale(self, context, alpha, out=None):
        GpuMatrix.wait_matrices(context, self)
        if out:
            out.last_modification_context = context
        else:
            self.last_modification_context = context
        context.activate()
        if out:
            gpu_matrix_kernels.scale(context.cuda_stream, self.nelems, alpha, self.data, out.data)
        else:
            cublas.s_scal(context.cublas_handle, self.nelems, alpha, self.data, 1)

    def assign_scaled_addition(self, context, alpha, a, b):
        """
        self = alpha * (a + b)
        """

        GpuMatrix.wait_matrices(context, a, b)
        self.last_modification_context = context
        context.activate()
        if a.nrows != b.nrows and a.ncols == b.ncols:
            raise ValueError('TODO!')
        gpu_matrix_kernels.assign_scaled_addition(context.cuda_stream, self.nelems, alpha, a.data, b.data, self.data)

    def assign_add(self, context, a, b):
        self.assign_scaled_addition(context, 1.0, a, b)

    def assign_scaled_subtraction(self, context, alpha, a, b):
        """
        self = alpha * (a - b)
        """

        GpuMatrix.wait_matrices(context, a, b)
        self.last_modification_context = context
        context.activate()
        if a.nrows != b.nrows and a.ncols == b.ncols:
            raise ValueError('TODO!')
        gpu_matrix_kernels.assign_scaled_subtraction(context.cuda_stream, self.nelems, alpha, a.data, b.data, self.data)

    def add_scaled_subtraction(self, context, alpha, a, b):
        """
        self += alpha * (a - b)
        """

        GpuMatrix.wait_matrices(context, a, b)
        self.last_modification_context = context
        context.activate()
        if a.nrows != b.nrows and a.ncols == b.ncols:
            raise ValueError('TODO!')
        gpu_matrix_kernels.add_scaled_subtraction(context.cuda_stream, self.nelems, alpha, a.data, b.data, self.data)

    def assign_sub(self, context, a, b):
        self.assign_scaled_addition(context, 1.0, a, b)

    # ========== TODO ============
    def add_scaled(self, context, alpha, a):
        """
        self += alpha * a
        """

        GpuMatrix.wait_matrices(context, a, self)
        self.last_modification_context = context
        context.activate()

        if self.nrows != 1 and a.nrows == 1:
            if self.ncols != a.ncols:
                raise ValueError('Operands could not be broadcast together with shapes ({},{}) ({},{})!'.format(self.nrows, self.ncols, a.nrows, a.ncols))
            gpu_matrix_kernels.matrix_vector_row_addition(context.cuda_stream, self.nrows, self.ncols, self.data, alpha, a.data, self.data)
        else:
            cublas.s_axpy(context.cublas_handle, self.nelems, alpha, a.data, 1, self.data, 1)

    def add(self, context, a):
        # TODO(sergii)
        if isinstance(a, GpuMatrix):
            self.add_scaled(context, ct.c_float(1.0), a)
        elif isinstance(a, quagga.matrix.SparseMatrix):
            GpuMatrix.add()
            GpuMatrix.sliced_columns_add()
            GpuMatrix.sliced_rows_batch_scaled_add()
        else:
            raise ValueError('TODO')

    def sub(self, context, a):
        self.add_scaled(context, ct.c_float(-1.0), a)
    # ============================

    def assign_sum(self, context, matrices):
        GpuMatrix.wait_matrices(context, *matrices)
        self.last_modification_context = context
        context.activate()

        n = len(matrices)
        matrices = (ct.POINTER(self.c_dtype) * n)(*(m.data for m in matrices))
        device_pointer = _get_temp_memory(context, n)
        elem_size = ct.sizeof(ct.POINTER(ct.c_float))
        cudart.cuda_memcpy_async(device_pointer, matrices, n * elem_size, 'default', context.cuda_stream)
        gpu_matrix_kernels.assign_sum(context.cuda_stream, self.nelems, device_pointer, n, self.data)

    def add_sum(self, context, matrices):
        GpuMatrix.wait_matrices(context, self, *matrices)
        self.last_modification_context = context
        context.activate()

        n = len(matrices)
        matrices = (ct.POINTER(self.c_dtype) * n)(*(m.data for m in matrices))
        device_pointer = _get_temp_memory(context, n)
        elem_size = ct.sizeof(ct.POINTER(ct.c_float))
        cudart.cuda_memcpy_async(device_pointer, matrices, n * elem_size, 'default', context.cuda_stream)
        gpu_matrix_kernels.add_sum(context.cuda_stream, self.nelems, device_pointer, n, self.data)

    def hprod(self, context, a):
        """
        self .*= a
        """

        GpuMatrix.wait_matrices(context, self, a)
        self.last_modification_context = context
        context.activate()

        if self.ncols != 1 and a.ncols == 1:
            if self.nrows != a.nrows:
                raise ValueError('Operands could not be broadcast together with shapes ({},{}) ({},{})!'.format(self.nrows, self.ncols, a.nrows, a.ncols))
            self.wait_matrices(context, self, a)
            context.activate()
            gpu_matrix_kernels.matrix_vector_column_hprod(context.cuda_stream, self.nrows, self.ncols, self.data, a.data, self.data)
        else:
            self.add_hprod(context, self, a, alpha=0.0)

    def add_hprod(self, context, a, b, c=None, alpha=1.0):
        """
        self = a .* b + alpha * self        or
        self = a .* b .* c + alpha * self
        """

        if c:
            GpuMatrix.wait_matrices(context, self, a, b, c)
        else:
            GpuMatrix.wait_matrices(context, self, a, b)
        self.last_modification_context = context
        context.activate()
        if not c:
            gpu_matrix_kernels.add_hadamard_product_2(context.cuda_stream, self.nelems, a.data, b.data, alpha, self.data)
        else:
            gpu_matrix_kernels.add_hadamard_product_3(context.cuda_stream, self.nelems, a.data, b.data, c.data, alpha, self.data)

    def assign_hprod(self, context, a, b, c=None):
        """
        self = a .* b       or
        self = a .* b .* c
        """

        if c:
            GpuMatrix.wait_matrices(context, a, b, c)
        else:
            GpuMatrix.wait_matrices(context, a, b)
        self.last_modification_context = context
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

        self.last_modification_context = context
        context.activate()
        if k is not None:
            GpuMatrix.wait_matrices(context, a, b, c, d, e, f, g, i, j, k)
            gpu_matrix_kernels.sum_hprod_11(context.cuda_stream, self.nelems, a.data, b.data, c.data, d.data, e.data, f.data, g.data, h.data, i.data, j.data, k.data, self.data)
        elif e is not None:
            GpuMatrix.wait_matrices(context, a, b, c, d, e)
            gpu_matrix_kernels.sum_hprod_5(context.cuda_stream, self.nelems, a.data, b.data, c.data, d.data, e.data, self.data)
        else:
            GpuMatrix.wait_matrices(context, a, b, c, d)
            gpu_matrix_kernels.sum_hprod_4(context.cuda_stream, self.nelems, a.data, b.data, c.data, d.data, self.data)

    def assign_hprod_sum(self, context, a, b):
        """
        self = sum(a .* b, axis=1)
        """
        GpuMatrix.wait_matrices(context, a, b)
        self.last_modification_context = context
        context.activate()
        gpu_matrix_kernels.hprod_sum(context.cuda_stream, a.nrows, a.ncols, a.data, b.data, self.data)

    def assign_dot(self, context, a, b, matrix_operation_a='N', matrix_operation_b='N'):
        self.add_dot(context, a, b, matrix_operation_a, matrix_operation_b, beta=ct.c_float(0.0))

    def add_dot(self, context, a, b, matrix_operation_a='N', matrix_operation_b='N', alpha=ct.c_float(1.0), beta=ct.c_float(1.0)):
        """
        self = alpha * op(a) * b + beta * self
        """

        if beta.value == 0.0:
            GpuMatrix.wait_matrices(context, a, b)
        else:
            GpuMatrix.wait_matrices(context, a, b, self)
        self.last_modification_context = context
        context.activate()
        if self.ncols == 1 and matrix_operation_b == 'N':
            cublas.s_gemv(context.cublas_handle, matrix_operation_a, a.nrows, a.ncols, alpha, a.data, a.nrows, b.data, 1, beta, self.data, 1)
        else:
            k = b.nrows if matrix_operation_b == 'N' else b.ncols
            cublas.s_gemm(context.cublas_handle, matrix_operation_a, matrix_operation_b, self.nrows, self.ncols, k, alpha, a.data, a.nrows, b.data, b.nrows, beta, self.data, self.nrows)


def _get_temp_memory(context, N):
    global __temp_pointer
    global __N
    pointer = __temp_pointer.get(context)
    if N > __N.get(context, -np.inf):
        if pointer:
            cudart.cuda_free(pointer)
        __N[context] = N + 10
        c_dtype = ct.POINTER(ct.c_float)
        elem_size = ct.sizeof(c_dtype)
        pointer = cudart.cuda_malloc(__N[context] * elem_size, c_dtype)
        __temp_pointer[context] = pointer
    return pointer


__temp_pointer = {}
__N = {}