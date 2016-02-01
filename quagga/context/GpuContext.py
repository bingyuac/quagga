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
from collections import defaultdict
from collections import deque

from quagga.cuda import cublas
from quagga.cuda import cudart
from quagga.cuda import cudnn
from quagga.utils import CustomDefaultDict


ct_py_object_p = ct.POINTER(ct.py_object)


def _create_disabled_timing_event():
    event = cudart.ct_cuda_event()
    cudart.cuda_event_create_with_flags(event, 'disable_timing')
    return event


def _create_cublas_handle(device_id):
    with cudart.device(device_id):
        handle = cublas.ct_cublas_handle()
        cublas.create(handle)
    return handle


def _create_cudnn_handle(device_id):
    with cudart.device(device_id):
        handle = cudnn.ct_cudnn_handle()
        cudnn.create(handle)
    return handle


def _create_callback(function):
    def callback(stream, status, user_data):
        cudart.check_cuda_status(status)
        args, kwargs = ct.cast(user_data, ct_py_object_p).contents.value
        function(*args, **kwargs)
        GpuContext._user_data[ct.cast(stream, ct.c_void_p).value].popleft()
    return cudart.ct_cuda_callback_type(callback)


class GpuContext(object):
    """
    Abstracts out low-level CUDA synchronisation primitives and standard library
    handles. Its methods provide inventory for high-level usage of CUDA streams,
    events, CUDNN handles and CUBLAS handles.

    Parameters
    ----------
    device_id : int
        Defines with which device the computational context will be associated.
    """
    _events = defaultdict(_create_disabled_timing_event)
    _cublas_handle = CustomDefaultDict(_create_cublas_handle)
    _cudnn_handle = CustomDefaultDict(_create_cudnn_handle)
    _user_data = defaultdict(deque)
    _callback_functions = CustomDefaultDict(_create_callback)

    def __init__(self, device_id=None):
        with cudart.device(device_id):
            self.device_id = cudart.cuda_get_device()
            self.cuda_stream = cudart.ct_cuda_stream()
            cudart.cuda_stream_create(self.cuda_stream)

    def __del__(self):
        with cudart.device(self.device_id):
            cudart.cuda_stream_destroy(self.cuda_stream)

    @property
    def cublas_handle(self):
        """
        Sets CUDA stream in CUBLAS handle and returns it.
        """
        cublas_handle = GpuContext._cublas_handle[self.device_id]
        cublas.set_stream(cublas_handle, self.cuda_stream)
        return cublas_handle

    @property
    def cudnn_handle(self):
        """
        Sets CUDA stream in CUDNN handle and returns it.
        """
        cudnn_handle = GpuContext._cudnn_handle[self.device_id]
        cudnn.set_stream(cudnn_handle, self.cuda_stream)
        return cudnn_handle

    def activate(self):
        """
        Activates the device associated with the context.
        """
        cudart.cuda_set_device(self.device_id)

    def synchronize(self):
        """
        Blocks the host until all preceding commands in the given context
        have completed.
        """
        cudart.cuda_stream_synchronize(self.cuda_stream)

    def wait(self, *args):
        """
        Makes all future work submitted to the context wait until all
        computations in ``args`` contexts have finished.

        Parameters
        ----------
        args : list of :class:`~quagga.context.GpuContext`

        """
        for context in args:
            context.activate()
            event = GpuContext._events[context, self]
            cudart.cuda_event_record(event, context.cuda_stream)
            self.activate()
            cudart.cuda_stream_wait_event(self.cuda_stream, event)

    def block(self, *args):
        """
        Makes all future work submitted to the ``args`` contexts wait until all
        computations in the context have finished.

        Parameters
        ----------
        args : list of :class:`~quagga.context.GpuContext`
        """
        for context in args:
            self.activate()
            event = GpuContext._events[self, context]
            cudart.cuda_event_record(event, self.cuda_stream)
            context.activate()
            cudart.cuda_stream_wait_event(context.cuda_stream, event)

    def add_callback(self, callback, *args, **kwargs):
        """
        Adds ``callback`` function to the current context, which will be called
        after all preceding computations have completed.

        Parameters
        ----------
        callback : python function
        args
            Arguments of the ``callback`` function.
        kwargs
            Named arguments of the ``callback`` function.
        """
        user_data = ct.py_object((args, kwargs))
        GpuContext._user_data[self.cuda_stream.value].append(user_data)
        cudart.\
            cuda_stream_add_callback(self.cuda_stream,
                                     GpuContext._callback_functions[callback],
                                     ct.byref(user_data))