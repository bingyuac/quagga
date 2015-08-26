"""
Python interface to CUDNN functions.
"""

import ctypes as ct
from quagga.cuda import cudart


_libcudnn = ct.cdll.LoadLibrary('libcudnn.so')


ct_cudnn_handle = ct.c_void_p
ct_cudnn_status = ct.c_int
ct_cudnn_tensor_descriptor = ct.c_void_p


cudnn_data_type = {
    'CUDNN_DATA_FLOAT': 0,
    'CUDNN_DATA_DOUBLE': 1
}


cudnn_tensor_format = {
    'CUDNN_TENSOR_NCHW': 0,
    'CUDNN_TENSOR_NHWC': 1
}


cudnn_softmax_algorithm = {
    'CUDNN_SOFTMAX_FAST': 0,
    'CUDNN_SOFTMAX_ACCURATE': 1
}


cudnn_softmax_mode = {
    'CUDNN_SOFTMAX_MODE_INSTANCE': 0,
    'CUDNN_SOFTMAX_MODE_CHANNEL': 1,
}


cudnn_statuses = {
    0: 'CUDNN_STATUS_SUCCESS',
    1: 'CUDNN_STATUS_NOT_INITIALIZED',
    2: 'CUDNN_STATUS_ALLOC_FAILED',
    3: 'CUDNN_STATUS_BAD_PARAM',
    4: 'CUDNN_STATUS_INTERNAL_ERROR',
    5: 'CUDNN_STATUS_INVALID_VALUE',
    6: 'CUDNN_STATUS_ARCH_MISMATCH',
    7: 'CUDNN_STATUS_MAPPING_ERROR',
    8: 'CUDNN_STATUS_EXECUTION_FAILED',
    9: 'CUDNN_STATUS_NOT_SUPPORTED',
    10: 'CUDNN_STATUS_LICENSE_ERROR',
}


class CudnnError(Exception):
    """CUDNN error."""
    pass


cudnn_exceptions = {}
for cudnn_error_code, cudnn_status_name in cudnn_statuses.iteritems():
    class_name = cudnn_status_name.replace('_STATUS_', '_')
    class_name = ''.join(each.capitalize() for each in class_name.split('_'))
    klass = type(class_name, (CudnnError, ), {'__doc__': cudnn_status_name})
    cudnn_exceptions[cudnn_error_code] = klass


def check_cudnn_status(status):
    if status != 0:
        try:
            raise cudnn_exceptions[status]
        except KeyError:
            raise CudnnError('unknown CUDNN error {}'.format(status))


_libcudnn.cudnnCreate.restype = ct_cudnn_status
_libcudnn.cudnnCreate.argtypes = [ct.POINTER(ct_cudnn_handle)]
def cudnn_create(handle):
    status = _libcudnn.cudnnCreate(ct.byref(handle))
    check_cudnn_status(status)


_libcudnn.cudnnDestroy.restype = ct_cudnn_status
_libcudnn.cudnnDestroy.argtypes = [ct_cudnn_handle]
def cudnn_destroy(handle):
    status = _libcudnn.cudnnDestroy(handle)
    check_cudnn_status(status)


_libcudnn.cudnnGetVersion.restype = ct.c_size_t
def cudnn_get_version():
    version = _libcudnn.cudnnGetVersion()
    return version


_libcudnn.cudnnSetStream.restype = ct_cudnn_status
_libcudnn.cudnnSetStream.argtypes = [ct_cudnn_handle, cudart.ct_cuda_stream]
def cudnn_set_stream(handle, stream):
    status = _libcudnn.cudnnSetStream(handle, stream)
    check_cudnn_status(status)


_libcudnn.cudnnGetStream.restype = ct_cudnn_status
_libcudnn.cudnnGetStream.argtypes = [ct_cudnn_handle, ct.POINTER(cudart.ct_cuda_stream)]
def cudnn_get_stream(handle, stream):
    status = _libcudnn.cudnnGetStream(handle, ct.byref(stream))
    check_cudnn_status(status)


_libcudnn.cudnnCreateTensorDescriptor.restype = ct_cudnn_status
_libcudnn.cudnnCreateTensorDescriptor.argtypes = [ct.POINTER(ct_cudnn_tensor_descriptor)]
def cudnn_create_tensor_descriptor(tensor_desc):
    status = _libcudnn.cudnnCreateTensorDescriptor(ct.byref(tensor_desc))
    check_cudnn_status(status)


_libcudnn.cudnnDestroyTensorDescriptor.restype = ct_cudnn_status
_libcudnn.cudnnDestroyTensorDescriptor.argtypes = [ct_cudnn_tensor_descriptor]
def cudnn_destroy_tensor_descriptor(tensor_desc):
    status = _libcudnn.cudnnDestroyTensorDescriptor(tensor_desc)
    check_cudnn_status(status)


_libcudnn.cudnnSetTensor4dDescriptor.restype = ct_cudnn_status
_libcudnn.cudnnSetTensor4dDescriptor.argtypes = [ct_cudnn_tensor_descriptor, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int]
def cudnn_set_tensor_4d_descriptor(tensor_desc, format, data_type, n, c, h, w):
    status = _libcudnn.cudnnSetTensor4dDescriptor(tensor_desc, format, data_type, n, c, h, w)
    check_cudnn_status(status)


_libcudnn.cudnnSetTensor4dDescriptorEx.restype = ct_cudnn_status
_libcudnn.cudnnSetTensor4dDescriptorEx.argtypes = [ct_cudnn_tensor_descriptor, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int]
def cudnn_set_tensor_4d_descriptor_ex(tensor_desc, data_type, n, c, h, w, n_stride, c_stride, h_stride, w_stride):
    status = _libcudnn.cudnnSetTensor4dDescriptorEx(tensor_desc, data_type, n, c, h, w, n_stride, c_stride, h_stride, w_stride)
    check_cudnn_status(status)


_libcudnn.cudnnSoftmaxForward.restype = ct_cudnn_status
_libcudnn.cudnnSoftmaxForward.argtypes = [ct_cudnn_handle, ct.c_int, ct.c_int, ct.c_void_p, ct_cudnn_tensor_descriptor, ct.c_void_p, ct.c_void_p, ct_cudnn_tensor_descriptor, ct.c_void_p]
def cudnn_softmax_forward(handle, algorithm, mode, alpha, src_desc, src_data, beta, dest_desc, dest_data):
    status = _libcudnn.cudnnSoftmaxForward(handle, algorithm, mode, ct.byref(alpha), src_desc, src_data, ct.byref(beta), dest_desc, dest_data)
    check_cudnn_status(status)