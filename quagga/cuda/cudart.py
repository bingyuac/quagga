"""
Python interface to CUDA runtime functions.
"""

import ctypes as ct
from contextlib import contextmanager


_libcudart = ct.cdll.LoadLibrary('libcudart.so')


ct_cuda_stream = ct.c_void_p
ct_cuda_event = ct.c_void_p
ct_cuda_error = ct.c_int


_libcudart.cudaGetErrorString.restype = ct.c_char_p
_libcudart.cudaGetErrorString.argtypes = [ct_cuda_error]
def cuda_get_error_string(e):
    """
    Retrieve CUDA error string.
    Return the string associated with the specified CUDA error status code.
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
    1: 'cudaErrorMissingConfiguration',
    2: 'cudaErrorMemoryAllocation',
    3: 'cudaErrorInitializationError',
    4: 'cudaErrorLaunchFailure',
    5: 'cudaErrorPriorLaunchFailure',
    6: 'cudaErrorLaunchTimeout',
    7: 'cudaErrorLaunchOutOfResources',
    8: 'cudaErrorInvalidDeviceFunction',
    9: 'cudaErrorInvalidConfiguration',
    10: 'cudaErrorInvalidDevice',
    11: 'cudaErrorInvalidValue',
    12: 'cudaErrorInvalidPitchValue',
    13: 'cudaErrorInvalidSymbol',
    14: 'cudaErrorMapBufferObjectFailed',
    15: 'cudaErrorUnmapBufferObjectFailed',
    16: 'cudaErrorInvalidHostPointer',
    17: 'cudaErrorInvalidDevicePointer',
    18: 'cudaErrorInvalidTexture',
    19: 'cudaErrorInvalidTextureBinding',
    20: 'cudaErrorInvalidChannelDescriptor',
    21: 'CudaErrorInvalidMemcpyDirection',
    22: 'CudaErrorAddressOfConstant',
    23: 'cudaErrorTextureFetchFailed',
    24: 'cudaErrorTextureNotBound',
    25: 'cudaErrorSynchronizationError',
    26: 'cudaErrorInvalidFilterSetting',
    27: 'cudaErrorInvalidNormSetting',
    28: 'cudaErrorMixedDeviceExecution',
    29: 'cudaErrorCudartUnloading',
    30: 'cudaErrorUnknown',
    31: 'cudaErrorNotYetImplemented',
    32: 'cudaErrorMemoryValueTooLarge',
    33: 'cudaErrorInvalidResourceHandle',
    34: 'cudaErrorNotReady',
    35: 'cudaErrorInsufficientDriver',
    36: 'cudaErrorSetOnActiveProcess',
    37: 'cudaErrorInvalidSurface',
    38: 'cudaErrorNoDevice',
    39: 'cudaErrorECCUncorrectable',
    40: 'cudaErrorSharedObjectSymbolNotFound',
    41: 'cudaErrorSharedObjectInitFailed',
    42: 'cudaErrorUnsupportedLimit',
    43: 'cudaErrorDuplicateVariableName',
    44: 'cudaErrorDuplicateTextureName',
    45: 'cudaErrorDuplicateSurfaceName',
    46: 'cudaErrorDevicesUnavailable',
    47: 'cudaErrorInvalidKernelImage',
    48: 'cudaErrorNoKernelImageForDevice',
    49: 'cudaErrorIncompatibleDriverContext',
    50: 'cudaErrorPeerAccessAlreadyEnabled',
    51: 'cudaErrorPeerAccessNotEnabled',
    54: 'cudaErrorDeviceAlreadyInUse',
    55: 'cudaErrorProfilerDisabled',
    56: 'cudaErrorProfilerNotInitialized',
    57: 'cudaErrorProfilerAlreadyStarted',
    58: 'cudaErrorProfilerAlreadyStopped',
    59: 'cudaErrorAssert',
    60: 'cudaErrorTooManyPeers',
    61: 'cudaErrorHostMemoryAlreadyRegistered',
    62: 'cudaErrorHostMemoryNotRegistered',
    63: 'cudaErrorOperatingSystem',
    64: 'cudaErrorPeerAccessUnsupported',
    65: 'cudaErrorLaunchMaxDepthExceeded',
    66: 'cudaErrorLaunchFileScopedTex',
    67: 'cudaErrorLaunchFileScopedSurf',
    68: 'cudaErrorSyncDepthExceeded',
    69: 'cudaErrorLaunchPendingCountExceeded',
    70: 'cudaErrorNotPermitted',
    71: 'cudaErrorNotSupported',
    72: 'cudaErrorHardwareStackError',
    73: 'cudaErrorIllegalInstruction',
    74: 'cudaErrorMisalignedAddress',
    75: 'cudaErrorInvalidAddressSpace',
    76: 'cudaErrorInvalidPc',
    77: 'cudaErrorIllegalAddress',
    78: 'cudaErrorInvalidPtx',
    79: 'cudaErrorInvalidGraphicsContext',
    127: 'cudaErrorStartupFailure',
    1000: 'cudaErrorApiFailureBase'
}


class CudaError(Exception):
    """CUDA error."""
    pass


cuda_exceptions = {}
for cuda_error_code, cuda_error_name in cuda_errors.iteritems():
    class_name = 'C' + cuda_error_name[1:]
    doc_string = cuda_get_error_string(cuda_error_code)
    klass = type(class_name, (CudaError, ), {'__doc__': doc_string})
    cuda_exceptions[cuda_error_code] = klass


def check_cuda_status(status):
    """
    Raise CUDA exception.
    Raise an exception corresponding to the specified CUDA runtime error code.
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
    Allocate memory on the device associated with the current active context.

    :param size: number of bytes of memory to allocate
    :param ctype: optional ctypes type to cast returned pointer.
    :return: pointer to allocated device memory.
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
    Free device memory.
    Free allocated memory on the device associated with the current active
    context.
    Parameters
    ----------
    ptr : ctypes pointer
        Pointer to allocated device memory.
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
    Copies count bytes from the memory area pointed to by src to the memory
    area pointed to by dst
    Parameters
    ----------
    dst : ctypes pointer
        Destination memory address
    src : ctypes pointer
        Source memory address
    count : int
        Size in bytes to copy
    kind: str
        Type of transfer
    """

    count = ct.c_size_t(count)
    status = _libcudart.cudaMemcpy(dst, src, count, cuda_memcpy_kinds[kind])
    check_cuda_status(status)


_libcudart.cudaMemcpyAsync.restype = ct_cuda_error
_libcudart.cudaMemcpyAsync.argtypes = [ct.c_void_p, ct.c_void_p,
                                       ct.c_size_t, ct.c_int, ct_cuda_stream]
def cuda_memcpy_async(dst, src, count, kind, stream):
    """
    Copies count bytes from the memory area pointed to by src to the memory
    area pointed to by dst
    Parameters
    ----------
    dst : ctypes pointer
        Destination memory address
    src : ctypes pointer
        Source memory address
    count : int
        Size in bytes to copy
    kind: str
        Type of transfer
    """

    count = ct.c_size_t(count)
    status = _libcudart.cudaMemcpyAsync(dst, src, count, cuda_memcpy_kinds[kind], stream)
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
    Return the amount of free and total device memory.
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
    :return:
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
    Set current CUDA device.
    Select a device to use for subsequent CUDA operations.
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
    Get current CUDA device.
    Return the identifying number of the device currently used to
    process CUDA operations.
    :return:
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
        Memory type
    device : int
        Number of device associated with pointer.
    is_managed : bool
        Indicates if the pointer ptr points to managed memory or not.
    """

    attributes = CudaPointerAttributes()
    status = _libcudart.cudaPointerGetAttributes(ct.byref(attributes), ptr)
    check_cuda_status(status)
    memory_type = cuda_memory_type[attributes.memoryType]
    return memory_type, attributes.device, bool(attributes.isManaged)


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


@contextmanager
def device(device_id):
    if device_id is None:
        yield
        return
    current_device_id = cuda_get_device()
    cuda_set_device(device_id)
    yield
    cuda_set_device(current_device_id)