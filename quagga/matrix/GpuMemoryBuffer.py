import ctypes as ct
from quagga.cuda import cudart
from quagga.matrix import GpuMatrix


class GpuMemoryBuffer(object):
    def __init__(self, device_id):
        self._device_id = device_id
        self._memory_pointer = None
        self._nbytes = 0
        self._offset = 0

    def extend_if_not_enough_space(self, matrices):
        if self._offset:
            raise ValueError('TODO')
        nbytes = sum(matrix.nbytes for matrix in matrices)
        if nbytes > self._nbytes:
            self.__del__()
            cudart.cuda_set_device(self._device_id)
            self._memory_pointer = cudart.cuda_malloc(nbytes)

    def get_matrix_copy(self, context, matrix):
        memory_pointer = ct.cast(self._memory_pointer.value + self._offset, ct.POINTER(matrix.c_dtype))
        a = GpuMatrix(memory_pointer, int(matrix.nrows), int(matrix.ncols), matrix.dtype, matrix.device_id, False)
        a.assign(context, matrix)
        self._offset += matrix.nbytes
        return a

    def clear(self):
        self._offset = 0

    def __del__(self):
        if self._memory_pointer:
            cudart.cuda_free(self._memory_pointer)