import atexit
from cuda import cudart, cublas
from collections import defaultdict


def _create_disabled_timing_event():
    event = cudart.cuda_event_t()
    cudart.cuda_event_create_with_flags(event, 'disable_timing')
    return event


class GpuMatrixContext(object):
    _events = defaultdict(_create_disabled_timing_event)
    _cublas_handle = None

    def __init__(self):
        if GpuMatrixContext._cublas_handle is None:
            GpuMatrixContext._cublas_handle = cublas.cublas_handle_t()
            cublas.cublas_create(GpuMatrixContext._cublas_handle)
        self.cuda_stream = cudart.cuda_stream_t()
        cudart.cuda_stream_create(self.cuda_stream)
        atexit.register(cudart.cuda_stream_destroy, self.cuda_stream)

    def __del__(self):
        cudart.cuda_stream_destroy(self.cuda_stream)
        atexit._exithandlers.remove((cudart.cuda_stream_destroy, (self.cuda_stream, ), {}))

    @property
    def cublas_handle(self):
        cublas.cublas_set_stream(GpuMatrixContext._cublas_handle, self.cuda_stream)
        return GpuMatrixContext._cublas_handle

    def synchronize(self):
        cudart.cuda_stream_synchronize(self.cuda_stream)

    def depend_on(self, *args):
        for context in args:
            event = GpuMatrixContext._events[context, self]
            cudart.cuda_event_record(event, context.cuda_stream)
            cudart.cuda_stream_wait_event(self.cuda_stream, event)

    def block(self, *args):
        for context in args:
            event = GpuMatrixContext._events[self, context]
            cudart.cuda_event_record(event, self.cuda_stream)
            cudart.cuda_stream_wait_event(context.cuda_stream, event)

    @staticmethod
    @atexit.register
    def __destroy_cublas_handle():
        if GpuMatrixContext._cublas_handle:
            cublas.cublas_destroy(GpuMatrixContext._cublas_handle)

    @staticmethod
    @atexit.register
    def __destroy_events():
        for event in GpuMatrixContext._events.itervalues():
            cudart.cuda_event_destroy(event)