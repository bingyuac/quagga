import quagga
from quagga.matrix import CpuMatrix, GpuMatrix


class Matrix(object):
    def __init__(self, *args, **kwargs):
        raise ValueError('Do not construct directly!')

    @classmethod
    def from_npa(cls, a, dtype=None, device_id=None):
        return cls._get_matrix_class().from_npa(a, dtype, device_id)

    @classmethod
    def empty(cls, nrows, ncols, dtype=None, device_id=None):
        dtype = dtype if dtype else quagga.dtype
        return cls._get_matrix_class().empty(nrows, ncols, dtype, device_id)

    @classmethod
    def empty_like(cls, other, device_id=None):
        return cls._get_matrix_class().empty_like(other, device_id)

    @staticmethod
    def _get_matrix_class():
        if quagga.processor_type == 'cpu':
            return CpuMatrix
        elif quagga.processor_type == 'gpu':
            return GpuMatrix
        else:
            raise ValueError(u'Processor type: {} is undefined'.
                             format(quagga.processor_type))