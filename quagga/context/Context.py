import quagga
from quagga.context import CpuContext, GpuContext


def Context():
    if quagga.processor_type == 'cpu':
        return CpuContext()
    elif quagga.processor_type == 'gpu':
        return GpuContext()
    else:
        raise ValueError(u'Processor type: {} is undefined'.
                         format(quagga.processor_type))