import quagga
from quagga.matrix import CpuMatrix, GpuMatrix


class MatrixType(type):
    def __getattr__(cls, name):
        return getattr(cls._get_matrix_class(), name)

    @staticmethod
    def _get_matrix_class():
        if quagga.processor_type == 'cpu':
            return CpuMatrix
        elif quagga.processor_type == 'gpu':
            return GpuMatrix
        else:
            raise ValueError(u'Processor type: {} is undefined'.
                             format(quagga.processor_type))


class Matrix(object):
    __metaclass__ = MatrixType

    def __init__(self, *args, **kwargs):
        raise ValueError('Do not construct directly!')