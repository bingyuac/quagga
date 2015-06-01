"""
Python interface to CUBLAS functions.
"""

import ctypes

from quagga.cuda import cudart

cublas_handle_t = ctypes.c_void_p
cublas_status_t = ctypes.c_int


_libcublas = ctypes.cdll.LoadLibrary('libcublas.so')


cublas_statuses = {
    1: 'CUBLAS_STATUS_NOT_INITIALIZED',
    3: 'CUBLAS_STATUS_ALLOC_FAILED',
    7: 'CUBLAS_STATUS_INVALID_VALUE',
    8: 'CUBLAS_STATUS_ARCH_MISMATCH',
    11: 'CUBLAS_STATUS_MAPPING_ERROR',
    13: 'CUBLAS_STATUS_EXECUTION_FAILED',
    14: 'CUBLAS_STATUS_INTERNAL_ERROR',
    15: 'CUBLAS_STATUS_NOT_SUPPORTED',
    16: 'CUBLAS_STATUS_LICENSE_ERROR'
}


class CublasError(Exception):
    """CUBLAS error."""
    pass


cublas_exceptions = {}
for cublas_error_code, cublas_status_name in cublas_statuses.iteritems():
    class_name = cublas_status_name.replace('_STATUS_', '_')
    class_name = ''.join(each.capitalize() for each in class_name.split('_'))
    klass = type(class_name, (CublasError, ), {'__doc__': cublas_status_name})
    cublas_exceptions[cublas_error_code] = klass


def check_cublas_status(status):
    if status != 0:
        try:
            raise cublas_exceptions[status]
        except KeyError:
            raise CublasError('unknown CUBLAS error {}'.format(status))


_libcublas.cublasCreate_v2.restype = cublas_status_t
_libcublas.cublasCreate_v2.argtypes = [ctypes.POINTER(cublas_handle_t)]
def cublas_create(handle):
    status = _libcublas.cublasCreate_v2(ctypes.byref(handle))
    check_cublas_status(status)


_libcublas.cublasDestroy_v2.restype = cublas_status_t
_libcublas.cublasDestroy_v2.argtypes = [cublas_handle_t]
def cublas_destroy(handle):
    status = _libcublas.cublasDestroy_v2(handle)
    check_cublas_status(status)


_libcublas.cublasGetVersion_v2.restype = cublas_status_t
_libcublas.cublasGetVersion_v2.argtypes = [cublas_handle_t, ctypes.c_void_p]
def cublas_get_version(handle):
    version = ctypes.c_int()
    status = _libcublas.cublasGetVersion_v2(handle, ctypes.byref(version))
    check_cublas_status(status)
    return version.value


_libcublas.cublasSetStream_v2.restype = cublas_status_t
_libcublas.cublasSetStream_v2.argtypes = [cublas_handle_t, cudart.cuda_stream_t]
def cublas_set_stream(handle, stream):
    status = _libcublas.cublasSetStream_v2(handle, stream)
    check_cublas_status(status)


_libcublas.cublasGetStream_v2.restype = cublas_status_t
_libcublas.cublasGetStream_v2.argtypes = [cublas_handle_t, ctypes.POINTER(
    cudart.cuda_stream_t)]
def cublas_get_stream(handle, stream):
    status = _libcublas.cublasGetStream_v2(handle, ctypes.byref(stream))
    check_cublas_status(status)


_libcublas.cublasSetVector.restype = cublas_status_t
_libcublas.cublasSetVector.argtypes = [ctypes.c_int, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_int]
def cublas_set_vector(n, elem_size, host_ptr, incx, device_ptr, incy):
    status = _libcublas.cublasSetVector(n, elem_size, host_ptr, incx, device_ptr, incy)
    check_cublas_status(status)


_libcublas.cublasSetVectorAsync.restype = cublas_status_t
_libcublas.cublasSetVectorAsync.argtypes = [ctypes.c_int, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_int,
                                            cudart.cuda_stream_t]
def cublas_set_vector_async(n, elem_size, host_ptr, incx, device_ptr, incy, stream):
    status = _libcublas.cublasSetVectorAsync(n, elem_size, host_ptr, incx, device_ptr, incy, stream)
    check_cublas_status(status)


_libcublas.cublasGetVector.restype = cublas_status_t
_libcublas.cublasGetVector.argtypes = [ctypes.c_int, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_int]
def cublas_get_vector(n, elem_size, device_ptr, incx, host_ptr, incy):
    status = _libcublas.cublasGetVector(n, elem_size, device_ptr, incx, host_ptr, incy)
    check_cublas_status(status)


_libcublas.cublasGetVectorAsync.restype = cublas_status_t
_libcublas.cublasGetVectorAsync.argtypes = [ctypes.c_int, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_int,
                                            cudart.cuda_stream_t]
def cublas_get_vector_async(n, elem_size, device_ptr, incx, host_ptr, incy, stream):
    status = _libcublas.cublasGetVectorAsync(n, elem_size, device_ptr, incx, host_ptr, incy, stream)
    check_cublas_status(status)


_libcublas.cublasSscal_v2.restype = cublas_status_t
_libcublas.cublasSscal_v2.argtypes = [cublas_handle_t, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_int]
def cublas_s_scal(handle, n, alpha, x, incx):
    status = _libcublas.cublasSscal_v2(handle, n, ctypes.byref(alpha), x, incx)
    check_cublas_status(status)


_libcublas.cublasSaxpy_v2.restype = cublas_status_t
_libcublas.cublasSaxpy_v2.argtypes = [cublas_handle_t, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_int]
def cublas_s_axpy(handle, n, alpha, x, incx, y, incy):
    status = _libcublas.cublasSaxpy_v2(handle, n, ctypes.byref(alpha), x, incx, y, incy)
    check_cublas_status(status)


_libcublas.cublasSdot_v2.restype = cublas_status_t
_libcublas.cublasSdot_v2.argtypes = [cublas_handle_t,
                                     ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_float),
                                     ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_float),
                                     ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_float)]
def cublas_s_dot(handle, n, x, incx, y, incy, result):
    status = _libcublas.cublasSdot_v2(handle, n, x, incx, y, incy, ctypes.byref(result))
    check_cublas_status(status)


cublas_op = {
    'n': 0,  # CUBLAS_OP_N
    'N': 0,
    't': 1,  # CUBLAS_OP_T
    'T': 1,
    'c': 2,  # CUBLAS_OP_C
    'C': 2,
}


_libcublas.cublasSgemv_v2.restype = cublas_status_t
_libcublas.cublasSgemv_v2.argtypes = [cublas_handle_t,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_int]
def cublas_s_gemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy):
    status = _libcublas.cublasSgemv_v2(handle, cublas_op[trans], m, n, ctypes.byref(alpha), a, lda, x, incx, beta, y, incy)
    check_cublas_status(status)


_libcublas.cublasSgemm_v2.restype = cublas_status_t
_libcublas.cublasSgemm_v2.argtypes = [cublas_handle_t,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_int]
def cublas_s_gemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc):
    status = _libcublas.cublasSgemm_v2(handle, cublas_op[transa], cublas_op[transb], m, n, k, ctypes.byref(alpha), a, lda, b, ldb, ctypes.byref(beta), c, ldc)
    check_cublas_status(status)
