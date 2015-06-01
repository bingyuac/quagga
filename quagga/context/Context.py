import quagga
from quagga.context import CpuContext, GpuContext


def Context():
    processor_type = quagga.config['processor_type']
    if processor_type == 'cpu':
        return CpuContext()
    elif processor_type == 'gpu':
        return GpuContext()
    else:
        raise ValueError(u'Processor type: {} is undefined'.format(processor_type))