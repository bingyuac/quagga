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
Python interface to CUSPARSE functions.
"""

import ctypes as ct
from quagga.cuda import cudart


_libcusparse = ct.cdll.LoadLibrary('libcusparse.so')


ct_cusparse_handle = ct.c_void_p
ct_cusparse_status = ct.c_int


cusparse_statuses = {
    0: 'CUSPARSE_STATUS_SUCCESS',
    1: 'CUSPARSE_STATUS_NOT_INITIALIZED',
    2: 'CUSPARSE_STATUS_ALLOC_FAILED',
    3: 'CUSPARSE_STATUS_INVALID_VALUE',
    4: 'CUSPARSE_STATUS_ARCH_MISMATCH',
    5: 'CUSPARSE_STATUS_MAPPING_ERROR',
    6: 'CUSPARSE_STATUS_EXECUTION_FAILED',
    7: 'CUSPARSE_STATUS_INTERNAL_ERROR',
    8: 'CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED',
    9: 'CUSPARSE_STATUS_ZERO_PIVOT'
}


class CublasError(Exception):
    """CUBLAS error."""
    pass


exceptions = {}
for error_code, status_name in cusparse_statuses.iteritems():
    class_name = status_name.replace('_STATUS_', '_')
    class_name = ''.join(each.capitalize() for each in class_name.split('_'))
    klass = type(class_name, (CublasError, ), {'__doc__': status_name})
    exceptions[error_code] = klass


def check_status(status):
    if status != 0:
        try:
            raise exceptions[status]
        except KeyError:
            raise CublasError('unknown CUSPARSE error {}'.format(status))

_libcusparse.cusparseCreate.restype = ct_cusparse_status
_libcusparse.cusparseCreate.argtypes = [ct.POINTER(ct_cusparse_handle)]
def create(handle):
    status = _libcusparse.cusparseCreate(ct.byref(handle))
    check_status(status)


_libcusparse.cusparseDestroy.restype = ct_cusparse_status
_libcusparse.cusparseDestroy.argtypes = [ct_cusparse_handle]
def destroy(handle):
    status = _libcusparse.cusparseDestroy(handle)
    check_status(status)


_libcusparse.cusparseGetVersion.restype = ct_cusparse_status
_libcusparse.cusparseGetVersion.argtypes = [ct_cusparse_handle, ct.c_void_p]
def get_version(handle):
    version = ct.c_int()
    status = _libcusparse.cusparseGetVersion(handle, ct.byref(version))
    check_status(status)
    return version.value


_libcusparse.cusparseSetStream.restype = ct_cusparse_status
_libcusparse.cusparseSetStream.argtypes = [ct_cusparse_handle,
                                           cudart.ct_cuda_stream]
def set_stream(handle, stream):
    status = _libcusparse.cusparseSetStream(handle, stream)
    check_status(status)