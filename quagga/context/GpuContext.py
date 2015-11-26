# ----------------------------------------------------------------------------
# Copyright 2015 Grammarly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import ctypes as ct
from collections import deque
from collections import defaultdict
from quagga.cuda import cudart, cublas, cudnn


ct_py_object_p = ct.POINTER(ct.py_object)


def _create_disabled_timing_event():
    event = cudart.ct_cuda_event()
    cudart.cuda_event_create_with_flags(event, 'disable_timing')
    return event


class GpuContext(object):
    _events = defaultdict(_create_disabled_timing_event)
    _cublas_handle = None
    _cudnn_handle = None
    _user_data = defaultdict(deque)

    def __init__(self, device_id=None):
        with cudart.device(device_id):
            self.device_id = cudart.cuda_get_device()
            self.cuda_stream = cudart.ct_cuda_stream()
            cudart.cuda_stream_create(self.cuda_stream)

    def __del__(self):
        cudart.cuda_stream_destroy(self.cuda_stream)

    @property
    def cublas_handle(self):
        cublas_handle = GpuContext._cublas_handle[self.device_id]
        cublas.set_stream(cublas_handle, self.cuda_stream)
        return cublas_handle

    @property
    def cudnn_handle(self):
        cudnn_handle = GpuContext._cudnn_handle[self.device_id]
        cudnn.set_stream(cudnn_handle, self.cuda_stream)
        return cudnn_handle

    def activate(self):
        cudart.cuda_set_device(self.device_id)

    def synchronize(self):
        cudart.cuda_stream_synchronize(self.cuda_stream)

    def wait(self, *args):
        """
        Makes all future work submitted to context wait until
        computations ends in `args` contexts
        """

        for context in args:
            context.activate()
            event = GpuContext._events[context, self]
            cudart.cuda_event_record(event, context.cuda_stream)
            self.activate()
            cudart.cuda_stream_wait_event(self.cuda_stream, event)

    def block(self, *args):
        for context in args:
            self.activate()
            event = GpuContext._events[self, context]
            cudart.cuda_event_record(event, self.cuda_stream)
            context.activate()
            cudart.cuda_stream_wait_event(context.cuda_stream, event)

    def add_callback(self, callback, *args, **kwargs):
        user_data = ct.py_object((args, kwargs))
        GpuContext._user_data[self.cuda_stream.value].append(user_data)
        cudart.cuda_stream_add_callback(self.cuda_stream, callback, ct.byref(user_data))

    @staticmethod
    def callback(function):
        def callback(stream, status, user_data):
            cudart.check_cuda_status(status)
            args, kwargs = ct.cast(user_data, ct_py_object_p).contents.value
            function(*args, **kwargs)
            GpuContext._user_data[ct.cast(stream, ct.c_void_p).value].popleft()
        return cudart.ct_cuda_callback_type(callback)


GpuContext._cublas_handle = []
GpuContext._cudnn_handle = []
for device_id in xrange(cudart.cuda_get_device_count()):
    with cudart.device(device_id):
        GpuContext._cublas_handle.append(cublas.ct_cublas_handle())
        cublas.create(GpuContext._cublas_handle[-1])
        GpuContext._cudnn_handle.append(cudnn.ct_cudnn_handle())
        cudnn.create(GpuContext._cudnn_handle[-1])