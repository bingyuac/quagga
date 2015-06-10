import atexit
from collections import defaultdict
from quagga.cuda import cudart, cublas


def _create_disabled_timing_event():
    event = cudart.ctypes_cuda_event()
    cudart.cuda_event_create_with_flags(event, 'disable_timing')
    return event


class GpuContext(object):
    _events = defaultdict(_create_disabled_timing_event)
    _cublas_handle = None

    def __init__(self):
        if GpuContext._cublas_handle is None:
            GpuContext._cublas_handle = cublas.ctypes_cublas_handle()
            cublas.cublas_create(GpuContext._cublas_handle)
            cublas.cublas_set_pointer_mode(GpuContext._cublas_handle, 'device')
        self.cuda_stream = cudart.ctypes_cuda_stream()
        cudart.cuda_stream_create(self.cuda_stream)
        atexit.register(cudart.cuda_stream_destroy, self.cuda_stream)

    def __del__(self):
        try:
            atexit._exithandlers.remove((cudart.cuda_stream_destroy, (self.cuda_stream, ), {}))
            cudart.cuda_stream_destroy(self.cuda_stream)
        except ValueError:
            pass

    @property
    def cublas_handle(self):
        cublas.cublas_set_stream(GpuContext._cublas_handle, self.cuda_stream)
        return GpuContext._cublas_handle

    def synchronize(self):
        cudart.cuda_stream_synchronize(self.cuda_stream)

    def depend_on(self, *args):
        for context in args:
            event = GpuContext._events[context, self]
            cudart.cuda_event_record(event, context.cuda_stream)
            cudart.cuda_stream_wait_event(self.cuda_stream, event)

    def block(self, *args):
        for context in args:
            event = GpuContext._events[self, context]
            cudart.cuda_event_record(event, self.cuda_stream)
            cudart.cuda_stream_wait_event(context.cuda_stream, event)

    @staticmethod
    @atexit.register
    def __destroy_cublas_handle():
        if GpuContext._cublas_handle:
            cublas.cublas_destroy(GpuContext._cublas_handle)

    @staticmethod
    @atexit.register
    def __destroy_events():
        for event in GpuContext._events.itervalues():
            cudart.cuda_event_destroy(event)