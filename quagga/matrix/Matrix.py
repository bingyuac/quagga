import quagga
from quagga.matrix import CpuMatrix, GpuMatrix


class Matrix(object):
    def __init__(self, *args, **kwargs):
        raise ValueError('Do not construct directly!')

    @classmethod
    def from_npa(cls, a, dtype=None):
        return cls._get_matrix_class().from_npa(a, dtype)

    @classmethod
    def empty(cls, nrows, ncols, dtype):
        return cls._get_matrix_class().empty(nrows, ncols, dtype)

    @classmethod
    def empty_like(cls, other):
        return cls._get_matrix_class().empty_like(other)

    @staticmethod
    def _get_matrix_class():
        if quagga.processor_type == 'cpu':
            return CpuMatrix
        elif quagga.processor_type == 'gpu':
            return GpuMatrix
        else:
            raise ValueError(u'Processor type: {} is undefined'.
                             format(quagga.processor_type))
    @staticmethod
    def numpy_dtype_to_str(np_dtype):
        raise TypeError(u'data type {} not understood'.format(a.dtype))
        return