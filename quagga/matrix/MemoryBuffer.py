import quagga
from quagga.matrix import CpuMemoryBuffer
from quagga.matrix import GpuMemoryBuffer


def MemoryBuffer(device_id):
    if quagga.processor_type == 'cpu':
        return CpuMemoryBuffer(device_id)
    elif quagga.processor_type == 'gpu':
        return GpuMemoryBuffer(device_id)
    else:
        raise ValueError(u'Processor type: {} is undefined'.
                         format(quagga.processor_type))