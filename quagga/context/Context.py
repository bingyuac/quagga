import quagga
from quagga.context import CpuContext
from quagga.context import GpuContext


def __get_context_class():
    if quagga.processor_type == 'cpu':
        return CpuContext
    elif quagga.processor_type == 'gpu':
        return GpuContext
    else:
        raise ValueError(u'Processor type: {} is undefined'.
                         format(quagga.processor_type))


def Context(device_id=None):
    return __get_context_class()(device_id)
Context.callback = lambda function: __get_context_class().callback(function)