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
"""
Python interface to CUDA runtime functions.
"""

import ctypes as ct
from contextlib import contextmanager


_libcudart = ct.cdll.LoadLibrary('libcudart.so')


ct_cuda_stream = ct.c_void_p
ct_cuda_event = ct.c_void_p
ct_cuda_error = ct.c_int
ct_cuda_callback_type = ct.CFUNCTYPE(None, ct_cuda_stream, ct_cuda_error, ct.c_void_p)


_libcudart.cudaGetErrorString.restype = ct.c_char_p
_libcudart.cudaGetErrorString.argtypes = [ct_cuda_error]
def cuda_get_error_string(e):
    """
    Retrieves CUDA error string.
    Returns a string associated with the specified CUDA error status code.

    Parameters
    ----------
    e : int
        Error number.

    Returns
    -------
    s : str
        Error string.
    """

    return _libcudart.cudaGetErrorString(e)


cuda_errors = {
    1: 'CudaErrorMissingConfiguration',
    2: 'CudaErrorMemoryAllocation',
    3: 'CudaErrorInitializationError',
    4: 'CudaErrorLaunchFailure',
    5: 'CudaErrorPriorLaunchFailure',
    6: 'CudaErrorLaunchTimeout',
    7: 'CudaErrorLaunchOutOfResources',
    8: 'CudaErrorInvalidDeviceFunction',
    9: 'CudaErrorInvalidConfiguration',
    10: 'CudaErrorInvalidDevice',
    11: 'CudaErrorInvalidValue',
    12: 'CudaErrorInvalidPitchValue',
    13: 'CudaErrorInvalidSymbol',
    14: 'CudaErrorMapBufferObjectFailed',
    15: 'CudaErrorUnmapBufferObjectFailed',
    16: 'CudaErrorInvalidHostPointer',
    17: 'CudaErrorInvalidDevicePointer',
    18: 'CudaErrorInvalidTexture',
    19: 'CudaErrorInvalidTextureBinding',
    20: 'CudaErrorInvalidChannelDescriptor',
    21: 'CudaErrorInvalidMemcpyDirection',
    22: 'CudaErrorAddressOfConstant',
    23: 'CudaErrorTextureFetchFailed',
    24: 'CudaErrorTextureNotBound',
    25: 'CudaErrorSynchronizationError',
    26: 'CudaErrorInvalidFilterSetting',
    27: 'CudaErrorInvalidNormSetting',
    28: 'CudaErrorMixedDeviceExecution',
    29: 'CudaErrorCudartUnloading',
    30: 'CudaErrorUnknown',
    31: 'CudaErrorNotYetImplemented',
    32: 'CudaErrorMemoryValueTooLarge',
    33: 'CudaErrorInvalidResourceHandle',
    34: 'CudaErrorNotReady',
    35: 'CudaErrorInsufficientDriver',
    36: 'CudaErrorSetOnActiveProcess',
    37: 'CudaErrorInvalidSurface',
    38: 'CudaErrorNoDevice',
    39: 'CudaErrorECCUncorrectable',
    40: 'CudaErrorSharedObjectSymbolNotFound',
    41: 'CudaErrorSharedObjectInitFailed',
    42: 'CudaErrorUnsupportedLimit',
    43: 'CudaErrorDuplicateVariableName',
    44: 'CudaErrorDuplicateTextureName',
    45: 'CudaErrorDuplicateSurfaceName',
    46: 'CudaErrorDevicesUnavailable',
    47: 'CudaErrorInvalidKernelImage',
    48: 'CudaErrorNoKernelImageForDevice',
    49: 'CudaErrorIncompatibleDriverContext',
    50: 'CudaErrorPeerAccessAlreadyEnabled',
    51: 'CudaErrorPeerAccessNotEnabled',
    54: 'CudaErrorDeviceAlreadyInUse',
    55: 'CudaErrorProfilerDisabled',
    56: 'CudaErrorProfilerNotInitialized',
    57: 'CudaErrorProfilerAlreadyStarted',
    58: 'CudaErrorProfilerAlreadyStopped',
    59: 'CudaErrorAssert',
    60: 'CudaErrorTooManyPeers',
    61: 'CudaErrorHostMemoryAlreadyRegistered',
    62: 'CudaErrorHostMemoryNotRegistered',
    63: 'CudaErrorOperatingSystem',
    64: 'CudaErrorPeerAccessUnsupported',
    65: 'CudaErrorLaunchMaxDepthExceeded',
    66: 'CudaErrorLaunchFileScopedTex',
    67: 'CudaErrorLaunchFileScopedSurf',
    68: 'CudaErrorSyncDepthExceeded',
    69: 'CudaErrorLaunchPendingCountExceeded',
    70: 'CudaErrorNotPermitted',
    71: 'CudaErrorNotSupported',
    72: 'CudaErrorHardwareStackError',
    73: 'CudaErrorIllegalInstruction',
    74: 'CudaErrorMisalignedAddress',
    75: 'CudaErrorInvalidAddressSpace',
    76: 'CudaErrorInvalidPc',
    77: 'CudaErrorIllegalAddress',
    78: 'CudaErrorInvalidPtx',
    79: 'CudaErrorInvalidGraphicsContext',
    127: 'CudaErrorStartupFailure',
    1000: 'CudaErrorApiFailureBase'
}


class CudaError(Exception):
    """CUDA error."""
    pass


cuda_exceptions = {}
for cuda_error_code, cuda_error_name in cuda_errors.iteritems():
    doc_string = cuda_get_error_string(cuda_error_code)
    klass = type(cuda_error_name, (CudaError, ), {'__doc__': doc_string})
    cuda_exceptions[cuda_error_code] = klass


def check_cuda_status(status):
    """
    Raises CUDA exception.
    Raises an exception corresponding to the specified CUDA runtime error code.

    Parameters
    ----------
    status : int
        CUDA runtime error code.
    """

    if status != 0:
        try:
            raise cuda_exceptions[status]
        except KeyError:
            raise CudaError('unknown CUDA error {}'.format(status))


_libcudart.cudaGetLastError.restype = ct_cuda_error
_libcudart.cudaGetLastError.argtypes = []
def cuda_get_last_error():
    return _libcudart.cudaGetLastError()


_libcudart.cudaMalloc.restype = ct_cuda_error
_libcudart.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
def cuda_malloc(size, ctype=None):
    """
    Allocates memory on the device associated with the current active CUDA
    context.

    Parameters
    ----------
    size : int
        Number of bytes of memory to allocate.
    ctype : ctypes type
        Optional ctypes type to cast returned pointer.

    Returns
    -------
    ptr : ctypes pointer
        Pointer to the allocated device memory.

    """
    ptr = ct.c_void_p()
    status = _libcudart.cudaMalloc(ct.byref(ptr), size)
    check_cuda_status(status)
    if ctype:
        ptr = ct.cast(ptr, ct.POINTER(ctype))
    return ptr


_libcudart.cudaFree.restype = ct_cuda_error
_libcudart.cudaFree.argtypes = [ct.c_void_p]
def cuda_free(ptr):
    """
    Deallocates device memory.
    Frees allocated memory on the device.

    Parameters
    ----------
    ptr : ctypes pointer
        Pointer to the allocated device memory.
    """

    status = _libcudart.cudaFree(ptr)
    check_cuda_status(status)


_libcudart.cudaMallocHost.restype = ct_cuda_error
_libcudart.cudaMallocHost.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
def cuda_malloc_host(size, ctype=None):
    ptr = ct.c_void_p()
    status = _libcudart.cudaMallocHost(ct.byref(ptr), size)
    check_cuda_status(status)
    if ctype:
        ptr = ct.cast(ptr, ct.POINTER(ctype))
    return ptr


cuda_memcpy_kinds = {
    'host_to_host': 0,
    'host_to_device': 1,
    'device_to_host': 2,
    'device_to_device': 3,
    'default': 4
}


_libcudart.cudaMemcpy.restype = ct_cuda_error
_libcudart.cudaMemcpy.argtypes = [ct.c_void_p, ct.c_void_p,
                                  ct.c_size_t, ct.c_int]
def cuda_memcpy(dst, src, count, kind):
    """
    Copies ``count`` bytes from the memory area pointed by ``src`` to the
    memory area pointed by ``dst``.

    Parameters
    ----------
    dst : ctypes pointer
        Destination memory address.
    src : ctypes pointer
        Source memory address.
    count : int
        Size in bytes to copy.
    kind: str
        Type of transfer.
    """

    count = ct.c_size_t(count)
    status = _libcudart.cudaMemcpy(dst, src, count, cuda_memcpy_kinds[kind])
    check_cuda_status(status)


_libcudart.cudaMemcpyAsync.restype = ct_cuda_error
_libcudart.cudaMemcpyAsync.argtypes = [ct.c_void_p, ct.c_void_p,
                                       ct.c_size_t, ct.c_int, ct_cuda_stream]
def cuda_memcpy_async(dst, src, count, kind, stream):
    """
    Copies ``count`` bytes from the memory area pointed by ``src`` to the
    memory area pointed by ``dst`` in the ``stream``.

    Parameters
    ----------
    dst : ctypes pointer
        Destination memory address.
    src : ctypes pointer
        Source memory address.
    count : int
        Size in bytes to copy.
    kind: str
        Type of transfer.
    stream: ct_cuda_stream
        Stream in which the data is being copied.
    """

    status = _libcudart.cudaMemcpyAsync(dst, src, count, cuda_memcpy_kinds[kind], stream)
    check_cuda_status(status)


_libcudart.cudaMemcpy2DAsync.restype = ct_cuda_error
_libcudart.cudaMemcpy2DAsync.argtypes = [ct.c_void_p, ct.c_size_t,
                                         ct.c_void_p, ct.c_size_t,
                                         ct.c_size_t, ct.c_size_t,
                                         ct.c_int, ct_cuda_stream]
def cuda_memcpy2d_async(dst, dpitch, src, spitch, width, height, kind, stream):
    status = _libcudart.cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cuda_memcpy_kinds[kind], stream)
    check_cuda_status(status)


_libcudart.cudaMemcpy2D.restype = ct_cuda_error
_libcudart.cudaMemcpy2D.argtypes = [ct.c_void_p, ct.c_size_t,
                                    ct.c_void_p, ct.c_size_t,
                                    ct.c_size_t, ct.c_size_t,
                                    ct.c_int]
def cuda_memcpy2d(dst, dpitch, src, spitch, width, height, kind):
    status = _libcudart.cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cuda_memcpy_kinds[kind])
    check_cuda_status(status)


_libcudart.cudaMemcpyPeer.restype = ct_cuda_error
_libcudart.cudaMemcpyPeer.argtypes = [ct.c_void_p, ct.c_int,
                                      ct.c_void_p, ct.c_int, ct.c_size_t]
def cuda_memcpy_peer(dst, dst_device, src, src_device, count):
    count = ct.c_size_t(count)
    status = _libcudart.cudaMemcpyPeer(dst, dst_device, src, src_device, count)
    check_cuda_status(status)


_libcudart.cudaMemcpyPeerAsync.restype = ct_cuda_error
_libcudart.cudaMemcpyPeerAsync.argtypes = [ct.c_void_p, ct.c_int,
                                           ct.c_void_p, ct.c_int,
                                           ct.c_size_t, ct_cuda_stream]
def cuda_memcpy_peer_async(dst, dst_device, src, src_device, count, stream):
    count = ct.c_size_t(count)
    status = _libcudart.cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream)
    check_cuda_status(status)


_libcudart.cudaMemGetInfo.restype = ct_cuda_error
_libcudart.cudaMemGetInfo.argtypes = [ct.POINTER(ct.c_size_t),
                                      ct.POINTER(ct.c_size_t)]
def cuda_mem_get_info():
    """
    Returns the amount of free and total device memory.

    Returns
    -------
    free : long
        Free memory in bytes.
    total : long
        Total memory in bytes.
    """

    free = ct.c_size_t()
    total = ct.c_size_t()
    status = _libcudart.cudaMemGetInfo(ct.byref(free), ct.byref(total))
    check_cuda_status(status)
    return free.value, total.value


_libcudart.cudaGetDeviceCount.restype = ct_cuda_error
_libcudart.cudaGetDeviceCount.argtypes = [ct.POINTER(ct.c_int)]
def cuda_get_device_count():
    """
    Returns the number of compute-capable devices.

    Returns
    -------
    count : int
        Number of compute-capable devices.
    """
    count = ct.c_int()
    status = _libcudart.cudaGetDeviceCount(ct.byref(count))
    check_cuda_status(status)
    return count.value


_libcudart.cudaSetDevice.restype = ct_cuda_error
_libcudart.cudaSetDevice.argtypes = [ct.c_int]
def cuda_set_device(device):
    """
    Sets current CUDA device.
    Selects a device for all subsequent CUDA operations to run on.

    Parameters
    ----------
    device : int
        Device number.
    """

    status = _libcudart.cudaSetDevice(device)
    check_cuda_status(status)


_libcudart.cudaGetDevice.restype = ct_cuda_error
_libcudart.cudaGetDevice.argtypes = [ct.POINTER(ct.c_int)]
def cuda_get_device():
    """
    Gets current CUDA device.
    Returns the id number of the device currently used for processing CUDA
    operations.

    Returns
    -------
    device : int
        Device number.
    """

    device = ct.c_int()
    status = _libcudart.cudaGetDevice(ct.byref(device))
    check_cuda_status(status)
    return device.value


_libcudart.cudaDriverGetVersion.restype = ct_cuda_error
_libcudart.cudaDriverGetVersion.argtypes = [ct.c_void_p]
def cuda_driver_get_version():
    """
    Get installed CUDA driver version.
    Return the version of the installed CUDA driver as an integer. If
    no driver is detected, 0 is returned.

    Returns
    -------
    version : int
        Driver version.
    """

    version = ct.c_int()
    status = _libcudart.cudaDriverGetVersion(ct.byref(version))
    check_cuda_status(status)
    return version.value


cuda_memory_type = {
    1: 'host',
    2: 'device'
}


class CudaPointerAttributes(ct.Structure):
    _fields_ = [
        ('memoryType', ct.c_int),
        ('device', ct.c_int),
        ('devicePointer', ct.c_void_p),
        ('hostPointer', ct.c_void_p),
        ('isManaged', ct.c_int)
        ]


_libcudart.cudaPointerGetAttributes.restype = ct_cuda_error
_libcudart.cudaPointerGetAttributes.argtypes = [ct.c_void_p,
                                                ct.c_void_p]
def cuda_pointer_get_attributes(ptr):
    """
    Get memory pointer attributes.
    Returns attributes of the specified pointer.

    Parameters
    ----------
    ptr : ctypes pointer
        Memory pointer to examine.

    Returns
    -------
    memory_type : str
        Memory type.
    device : int
        Number of device associated with pointer.
    is_managed : bool
        Indicates if the pointer ``ptr`` points to managed memory or not.
    """

    attributes = CudaPointerAttributes()
    status = _libcudart.cudaPointerGetAttributes(ct.byref(attributes), ptr)
    check_cuda_status(status)
    memory_type = cuda_memory_type[attributes.memoryType]
    return memory_type, attributes.device, bool(attributes.isManaged)


_libcudart.cudaDeviceSynchronize.restype = ct_cuda_error
_libcudart.cudaDeviceSynchronize.argtypes = []
def cuda_device_synchronize():
    status = _libcudart.cudaDeviceSynchronize()
    check_cuda_status(status)


_libcudart.cudaStreamCreate.restype = ct_cuda_error
_libcudart.cudaStreamCreate.argtypes = [ct.POINTER(ct_cuda_stream)]
def cuda_stream_create(stream):
    status = _libcudart.cudaStreamCreate(ct.byref(stream))
    check_cuda_status(status)


_libcudart.cudaStreamDestroy.restype = ct_cuda_error
_libcudart.cudaStreamDestroy.argtypes = [ct_cuda_stream]
def cuda_stream_destroy(stream):
    status = _libcudart.cudaStreamDestroy(stream)
    check_cuda_status(status)


_libcudart.cudaStreamSynchronize.restype = ct_cuda_error
_libcudart.cudaStreamSynchronize.argtypes = [ct_cuda_stream]
def cuda_stream_synchronize(stream):
    status = _libcudart.cudaStreamSynchronize(stream)
    check_cuda_status(status)


_libcudart.cudaStreamWaitEvent.restype = ct_cuda_error
_libcudart.cudaStreamWaitEvent.argtypes = [ct_cuda_stream, ct_cuda_event, ct.c_uint]
def cuda_stream_wait_event(stream, event):
    status = _libcudart.cudaStreamWaitEvent(stream, event, 0)
    check_cuda_status(status)


_libcudart.cudaStreamAddCallback.restype = ct_cuda_error
_libcudart.cudaStreamAddCallback.argtypes = [ct_cuda_stream, ct_cuda_callback_type, ct.c_void_p, ct.c_uint]
def cuda_stream_add_callback(stream, callback, user_data):
    _libcudart.cudaStreamAddCallback(stream, callback, user_data, 0)


cuda_event_flag = {
    'default': 0,
    'blocking_sync': 1,
    'disable_timing': 2,
    'interprocess': 4
}


_libcudart.cudaEventCreate.restype = ct_cuda_error
_libcudart.cudaEventCreate.argtypes = [ct.POINTER(ct_cuda_event)]
def cuda_event_create(event):
    status = _libcudart.cudaEventCreate(ct.byref(event))
    check_cuda_status(status)


_libcudart.cudaEventCreateWithFlags.restype = ct_cuda_error
_libcudart.cudaEventCreateWithFlags.argtypes = [ct.POINTER(ct_cuda_event),
                                                ct.c_uint]
def cuda_event_create_with_flags(event, flag):
    status = _libcudart.cudaEventCreateWithFlags(ct.byref(event), cuda_event_flag[flag])
    check_cuda_status(status)


_libcudart.cudaEventDestroy.restype = ct_cuda_error
_libcudart.cudaEventDestroy.argtypes = [ct_cuda_event]
def cuda_event_destroy(event):
    status = _libcudart.cudaEventDestroy(event)
    check_cuda_status(status)


_libcudart.cudaEventRecord.restype = ct_cuda_error
_libcudart.cudaEventRecord.argtypes = [ct_cuda_event, ct_cuda_stream]
def cuda_event_record(event, stream):
    status = _libcudart.cudaEventRecord(event, stream)
    check_cuda_status(status)


_libcudart.cudaDeviceReset.restype = ct_cuda_error
def cuda_device_reset():
    status = _libcudart.cudaDeviceReset()
    check_cuda_status(status)


cuda_device_flag = {
    'cuda_device_schedule_auto': 0,
    'cuda_device_schedule_spin': 1,
    'cuda_device_schedule_yield': 2,
    'cuda_device_schedule_blocking_sync': 4
}


_libcudart.cudaSetDeviceFlags.restype = ct_cuda_error
_libcudart.cudaSetDeviceFlags.argtypes = [ct.c_uint]
def cuda_set_device_flags(flag):
    status = _libcudart.cudaSetDeviceFlags(cuda_device_flag[flag])
    check_cuda_status(status)


class CudaIpcMemHandleType(ct.Structure):
    _fields_ = [('reserved', ct.c_char * 64)]


_libcudart.cudaIpcGetMemHandle.restype = ct_cuda_error
_libcudart.cudaIpcGetMemHandle.argtypes = [ct.c_void_p, ct.c_void_p]
def cuda_ipc_get_mem_handle(handle, ptr):
    status = _libcudart.cudaIpcGetMemHandle(ct.byref(handle), ptr)
    check_cuda_status(status)


_libcudart.cudaIpcOpenMemHandle.restype = ct_cuda_error
_libcudart.cudaIpcOpenMemHandle.argtypes = [ct.POINTER(ct.c_void_p),
                                            CudaIpcMemHandleType,
                                            ct.c_int]
def cuda_ipc_open_mem_handle(ptr, handle):
    status = _libcudart.cudaIpcOpenMemHandle(ct.byref(ptr), handle, 1)
    check_cuda_status(status)


_libcudart.cudaIpcCloseMemHandle.restype = ct_cuda_error
_libcudart.cudaIpcCloseMemHandle.argtypes = [ct.c_void_p]
def cuda_ipc_close_mem_handle(ptr):
    status = _libcudart.cudaIpcCloseMemHandle(ptr)
    check_cuda_status(status)


_libcudart.cudaDeviceEnablePeerAccess.restype = ct_cuda_error
_libcudart.cudaDeviceEnablePeerAccess.argtypes = [ct.c_int, ct.c_uint]
def cuda_device_enable_peer_access(peer_device):
    status = _libcudart.cudaDeviceEnablePeerAccess(peer_device, 0)
    check_cuda_status(status)


_libcudart.cudaDeviceCanAccessPeer.restype = ct_cuda_error
_libcudart.cudaDeviceCanAccessPeer.argtypes = [ct.POINTER(ct.c_int), ct.c_int, ct.c_uint]
def cuda_device_can_access_peer(device, peer_device):
    can_access_peer = ct.c_int()
    status = _libcudart.cudaDeviceCanAccessPeer(ct.byref(can_access_peer), device, peer_device)
    check_cuda_status(status)
    return can_access_peer


@contextmanager
def device(device_id):
    if device_id is None:
        yield
        return
    current_device_id = cuda_get_device()
    cuda_set_device(device_id)
    yield
    cuda_set_device(current_device_id)