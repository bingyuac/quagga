"""
Python interface to CUDA runtime functions.
"""

import ctypes


cuda_stream_t = ctypes.c_void_p
cuda_event_t = ctypes.c_void_p
cuda_error_t = ctypes.c_int


_libcudart = ctypes.cdll.LoadLibrary('libcudart.so')


_libcudart.cudaGetErrorString.restype = ctypes.c_char_p
_libcudart.cudaGetErrorString.argtypes = [cuda_error_t]
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
    See Also
    --------
    cuda_exceptions
    """

    if status != 0:
        try:
            raise cuda_exceptions[status]
        except KeyError:
            raise CudaError('unknown CUDA error {}'.format(status))


_libcudart.cudaGetLastError.restype = cuda_error_t
_libcudart.cudaGetLastError.argtypes = []
def cuda_get_last_error():
    return _libcudart.cudaGetLastError()


_libcudart.cudaMalloc.restype = cuda_error_t
_libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.c_size_t]
def cuda_malloc(size, ctype=None):
    """
    Allocate device memory.
    Allocate memory on the device associated with the current active context.
    Parameters
    ----------
    count : int
        Number of bytes of memory to allocate
    ctype : ctypes._SimpleCDat, optional
        ctypes type to cast returned pointer.
    Returns
    -------
    ptr : ctypes pointer
        Pointer to allocated device memory.
    """

    ptr = ctypes.c_void_p()
    status = _libcudart.cudaMalloc(ctypes.byref(ptr), size)
    check_cuda_status(status)
    if ctype:
        ptr = ctypes.cast(ptr, ctypes.POINTER(ctype))
    return ptr


_libcudart.cudaFree.restype = cuda_error_t
_libcudart.cudaFree.argtypes = [ctypes.c_void_p]
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


cuda_memcpy_kinds = {
    'host_to_host': 0,
    'host_to_device': 1,
    'device_to_host': 2,
    'device_to_device': 3,
    'default': 4
}


_libcudart.cudaMemcpy.restype = cuda_error_t
_libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int]
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

    count = ctypes.c_size_t(count)
    status = _libcudart.cudaMemcpy(dst, src, count, cuda_memcpy_kinds[kind])
    check_cuda_status(status)


_libcudart.cudaMemcpyAsync.restype = cuda_error_t
_libcudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t, ctypes.c_int]
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

    count = ctypes.c_size_t(count)
    status = _libcudart.cudaMemcpyAsync(dst, src, count, cuda_memcpy_kinds[kind], stream)
    check_cuda_status(status)


_libcudart.cudaMemGetInfo.restype = cuda_error_t
_libcudart.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                                      ctypes.POINTER(ctypes.c_size_t)]
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

    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    status = _libcudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    check_cuda_status(status)
    return free.value, total.value


_libcudart.cudaSetDevice.restype = cuda_error_t
_libcudart.cudaSetDevice.argtypes = [ctypes.c_int]
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


_libcudart.cudaGetDevice.restype = cuda_error_t
_libcudart.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
def cuda_get_device():
    """
    Get current CUDA device.
    Return the identifying number of the device currently used to
    process CUDA operations.
    Returns
    -------
    device : int
        Device number.
    """

    device = ctypes.c_int()
    status = _libcudart.cudaGetDevice(ctypes.byref(device))
    check_cuda_status(status)
    return device.value


_libcudart.cudaDriverGetVersion.restype = cuda_error_t
_libcudart.cudaDriverGetVersion.argtypes = [ctypes.c_void_p]
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

    version = ctypes.c_int()
    status = _libcudart.cudaDriverGetVersion(ctypes.byref(version))
    check_cuda_status(status)
    return version.value


cuda_memory_type = {
    1: 'host',
    2: 'device'
}


class CudaPointerAttributes(ctypes.Structure):
    _fields_ = [
        ('memoryType', ctypes.c_int),
        ('device', ctypes.c_int),
        ('devicePointer', ctypes.c_void_p),
        ('hostPointer', ctypes.c_void_p),
        ('isManaged', ctypes.c_int)
        ]


_libcudart.cudaPointerGetAttributes.restype = cuda_error_t
_libcudart.cudaPointerGetAttributes.argtypes = [ctypes.c_void_p,
                                                ctypes.c_void_p]
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
    status = _libcudart.cudaPointerGetAttributes(ctypes.byref(attributes), ptr)
    check_cuda_status(status)
    memory_type = cuda_memory_type[attributes.memoryType]
    return memory_type, attributes.device, bool(attributes.isManaged)


_libcudart.cudaStreamCreate.restype = cuda_error_t
_libcudart.cudaStreamCreate.argtypes = [ctypes.POINTER(cuda_stream_t)]
def cuda_stream_create(stream):
    status = _libcudart.cudaStreamCreate(ctypes.byref(stream))
    check_cuda_status(status)


_libcudart.cudaStreamDestroy.restype = cuda_error_t
_libcudart.cudaStreamDestroy.argtypes = [cuda_stream_t]
def cuda_stream_destroy(stream):
    status = _libcudart.cudaStreamDestroy(stream)
    check_cuda_status(status)


_libcudart.cudaStreamSynchronize.restype = cuda_error_t
_libcudart.cudaStreamSynchronize.argtypes = [cuda_stream_t]
def cuda_stream_synchronize(stream):
    status = _libcudart.cudaStreamSynchronize(stream)
    check_cuda_status(status)


_libcudart.cudaStreamWaitEvent.restype = cuda_error_t
_libcudart.cudaStreamWaitEvent.argtypes = [cuda_stream_t, cuda_event_t, ctypes.c_uint]
def cuda_stream_wait_event(stream, event):
    status = _libcudart.cudaStreamWaitEvent(stream, event, 0)
    check_cuda_status(status)


cuda_event_flag = {
    'default': 0,
    'blocking_sync': 1,
    'disable_timing': 2,
    'interprocess': 4
}


_libcudart.cudaEventCreate.restype = cuda_error_t
_libcudart.cudaEventCreate.argtypes = [ctypes.POINTER(cuda_event_t)]
def cuda_event_create(event):
    status = _libcudart.cudaEventCreate(ctypes.byref(event))
    check_cuda_status(status)


_libcudart.cudaEventCreateWithFlags.restype = cuda_error_t
_libcudart.cudaEventCreateWithFlags.argtypes = [ctypes.POINTER(cuda_event_t),
                                                ctypes.c_uint]
def cuda_event_create_with_flags(event, flag):
    status = _libcudart.cudaEventCreateWithFlags(ctypes.byref(event), cuda_event_flag[flag])
    check_cuda_status(status)


_libcudart.cudaEventDestroy.restype = cuda_error_t
_libcudart.cudaEventDestroy.argtypes = [cuda_event_t]
def cuda_event_destroy(event):
    status = _libcudart.cudaEventDestroy(event)
    check_cuda_status(status)


_libcudart.cudaEventRecord.restype = cuda_error_t
_libcudart.cudaEventRecord.argtypes = [cuda_event_t, cuda_stream_t]
def cuda_event_record(event, stream):
    status = _libcudart.cudaEventRecord(event, stream)
    check_cuda_status(status)


_libcudart.cudaDeviceReset.restype = cuda_error_t
def cuda_device_reset():
    status = _libcudart.cudaDeviceReset()
    check_cuda_status(status)


cuda_device_flag = {
    'cuda_device_schedule_auto': 0,
    'cuda_device_schedule_spin': 1,
    'cuda_device_schedule_yield': 2,
    'cuda_device_schedule_blocking_sync': 4
}


_libcudart.cudaSetDeviceFlags.restype = cuda_error_t
_libcudart.cudaSetDeviceFlags.argtypes = [ctypes.c_uint]
def cuda_set_device_flags(flag):
    status = _libcudart.cudaSetDeviceFlags(cuda_device_flag[flag])
    check_cuda_status(status)