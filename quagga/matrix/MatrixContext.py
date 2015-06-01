import quagga
from quagga.matrix import CpuMatrixContext, GpuMatrixContext


def MatrixContext():
    processor_type = quagga.config['processor_type']
    if processor_type == 'cpu':
        return CpuMatrixContext()
    elif processor_type == 'gpu':
        return GpuMatrixContext()
    else:
        raise ValueError(u'Processor type: {} is undefined'.format(processor_type))