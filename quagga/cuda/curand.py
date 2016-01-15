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
Python interface to CURAND functions.
"""

import ctypes as ct
from quagga.cuda import cudart


_libcurand = ct.cdll.LoadLibrary('libcurand.so')


ct_curand_generator = ct.c_void_p
ct_curand_status = ct.c_int


curand_rng_type = {
    'CURAND_RNG_TEST': 0,
    'CURAND_RNG_PSEUDO_DEFAULT': 100,
    'CURAND_RNG_PSEUDO_XORWOW': 101,
    'CURAND_RNG_PSEUDO_MRG32K3A': 121,
    'CURAND_RNG_PSEUDO_MTGP32': 141,
    'CURAND_RNG_PSEUDO_MT19937': 142,
    'CURAND_RNG_PSEUDO_PHILOX4_32_10': 161,
    'CURAND_RNG_QUASI_DEFAULT': 200,
    'CURAND_RNG_QUASI_SOBOL32': 201,
    'CURAND_RNG_QUASI_SCRAMBLED_SOBOL32': 202,
    'CURAND_RNG_QUASI_SOBOL64': 203,
    'CURAND_RNG_QUASI_SCRAMBLED_SOBOL64': 204
}


curand_statuses = {
    0: 'CUDNN_STATUS_SUCCESS',
    100: 'CURAND_STATUS_VERSION_MISMATCH',
    101: 'CURAND_STATUS_NOT_INITIALIZED',
    102: 'CURAND_STATUS_ALLOCATION_FAILED',
    103: 'CURAND_STATUS_TYPE_ERROR',
    104: 'CURAND_STATUS_OUT_OF_RANGE',
    105: 'CURAND_STATUS_LENGTH_NOT_MULTIPLE',
    106: 'CURAND_STATUS_DOUBLE_PRECISION_REQUIRED',
    201: 'CURAND_STATUS_LAUNCH_FAILURE',
    202: 'CURAND_STATUS_PREEXISTING_FAILURE',
    203: 'CURAND_STATUS_INITIALIZATION_FAILED',
    204: 'CURAND_STATUS_ARCH_MISMATCH',
    999: 'CURAND_STATUS_INTERNAL_ERROR'
}


class CurandError(Exception):
    """CURAND error."""
    pass


exceptions = {}
for error_code, status_name in curand_statuses.iteritems():
    class_name = status_name.replace('_STATUS_', '_')
    class_name = ''.join(each.capitalize() for each in class_name.split('_'))
    klass = type(class_name, (CurandError, ), {'__doc__': status_name})
    exceptions[error_code] = klass


def check_status(status):
    if status != 0:
        try:
            raise exceptions[status]
        except KeyError:
            raise CurandError('unknown CURAND error {}'.format(status))


_libcurand.curandCreateGenerator.restype = ct_curand_status
_libcurand.curandCreateGenerator.argtypes = [ct.POINTER(ct_curand_generator),
                                             ct.c_int]
def create_generator(generator, rng_type):
    """
    TODO
    Parameters
    ----------
    generator
    rng_type

    Returns
    -------

    """
    status = _libcurand.curandCreateGenerator(ct.byref(generator), rng_type)
    check_status(status)


_libcurand.curandDestroyGenerator.restype = ct_curand_status
_libcurand.curandDestroyGenerator.argtypes = [ct_curand_generator, ct.c_int]
def destroy_generator(generator):
    """
    TODO
    Parameters
    ----------
    generator

    Returns
    -------

    """
    status = _libcurand.curandDestroyGenerator(generator)
    check_status(status)


_libcurand.curandSetPseudoRandomGeneratorSeed.restype = ct_curand_status
_libcurand.curandSetPseudoRandomGeneratorSeed.argtypes = [ct_curand_generator,
                                                          ct.c_ulonglong]
def pseudo_random_generator_seed(generator, seed):
    status = _libcurand.curandSetPseudoRandomGeneratorSeed(generator, seed)
    check_status(status)


_libcurand.curandGenerateUniform.restype = ct_curand_status
_libcurand.curandGenerateUniform.argtypes = [ct_curand_generator,
                                             ct.POINTER(ct.c_float),
                                             ct.c_size_t]
def generate_uniform(generator, output_ptr, num):
    status = _libcurand.curandGenerateUniform(generator, output_ptr, num)
    check_status(status)


_libcurand.curandGenerateNormal.restype = ct_curand_status
_libcurand.curandGenerateNormal.argtypes = [ct_curand_generator,
                                            ct.POINTER(ct.c_float),
                                            ct.c_size_t,
                                            ct.c_float,
                                            ct.c_float]
def generate_normal(generator, output_ptr, n, mean, stddev):
    status = _libcurand.curandGenerateNormal(generator, output_ptr, n, mean, stddev)
    check_status(status)


_libcurand.curandSetStream.restype = ct_curand_status
_libcurand.curandSetStream.argtypes = [ct_curand_generator,
                                       cudart.ct_cuda_stream]
def set_stream(generator, stream):
    status = _libcurand.curandSetStream(generator, stream)
    check_status(status)