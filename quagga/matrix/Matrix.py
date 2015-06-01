import quagga
from quagga.matrix import CpuMatrix, GpuMatrix


class Matrix(object):
    def __init__(self, *args, **kwargs):
        raise ValueError('Do not construct directly!')

    @classmethod
    def from_npa(cls, a):
        return cls._get_matrix_class().from_npa(a)

    @classmethod
    def empty(cls, nrows, ncols):
        return cls._get_matrix_class().empty(nrows, ncols)

    @classmethod
    def empty_like(cls, other):
        return cls._get_matrix_class().empty_like(other)

    @staticmethod
    def _get_matrix_class():
        processor_type = quagga.config['processor_type']
        if processor_type == 'cpu':
            return CpuMatrix
        elif processor_type == 'gpu':
            return GpuMatrix
        else:
            raise ValueError(u'Processor type: {} is undefined'.format(processor_type))