import quagga
from quagga.context import CpuContext, GpuContext


def Context(device_id):
    if quagga.processor_type == 'cpu':
        return CpuContext(device_id)
    elif quagga.processor_type == 'gpu':
        return GpuContext(device_id)
    else:
        raise ValueError(u'Processor type: {} is undefined'.
                         format(quagga.processor_type))