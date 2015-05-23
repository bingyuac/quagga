import ctypes
from cuda import cudart


# gpu_matrix_kernels = ctypes.cdll.LoadLibrary('gpu_matrix_kernels.so')
gpu_matrix_kernels = ctypes.cdll.LoadLibrary('/home/sergii/Desktop/GrammarticalityJudge/c_lib/gpu_matrix_kernels.so')


gpu_matrix_kernels._scale.restype = cudart.cuda_error_t
gpu_matrix_kernels._scale.argtypes = [cudart.cuda_stream_t,
                                      ctypes.c_int,
                                      ctypes.c_float,
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float)]
def scale(stream, nelems, alpha, data, out_data):
    status = gpu_matrix_kernels._scale(stream, nelems, alpha, data, out_data)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._tanh.restype = cudart.cuda_error_t
gpu_matrix_kernels._tanh.argtypes = [cudart.cuda_stream_t,
                                     ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_float),
                                     ctypes.POINTER(ctypes.c_float)]
def tanh(stream, nelems, data, tanh_data):
    status = gpu_matrix_kernels._tanh(stream, nelems, data, tanh_data)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._tanh_der.restype = cudart.cuda_error_t
gpu_matrix_kernels._tanh_der.argtypes = [cudart.cuda_stream_t,
                                         ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_float)]
def tanh_der(stream, nelems, data, tanh_data, derivative):
    status = gpu_matrix_kernels._tanh_der(stream, nelems, data, tanh_data, derivative)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._sigmoid.restype = cudart.cuda_error_t
gpu_matrix_kernels._sigmoid.argtypes = [cudart.cuda_stream_t,
                                        ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_float),
                                        ctypes.POINTER(ctypes.c_float)]
def sigmoid(stream, nelems, data, sigmoid_data):
    status = gpu_matrix_kernels._sigmoid(stream, nelems, data, sigmoid_data)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._sigmoid_der.restype = cudart.cuda_error_t
gpu_matrix_kernels._sigmoid_der.argtypes = [cudart.cuda_stream_t,
                                            ctypes.c_int,
                                            ctypes.POINTER(ctypes.c_float),
                                            ctypes.POINTER(ctypes.c_float),
                                            ctypes.POINTER(ctypes.c_float)]
def sigmoid_der(stream, nelems, data, sigmoid_data, derivative):
    status = gpu_matrix_kernels._sigmoid_der(stream, nelems, data, sigmoid_data, derivative)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._sum.restype = cudart.cuda_error_t
gpu_matrix_kernels._sum.argtypes = [cudart.cuda_stream_t,
                                    ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_float),
                                    ctypes.POINTER(ctypes.c_float),
                                    ctypes.POINTER(ctypes.c_float),
                                    ctypes.POINTER(ctypes.c_float),
                                    ctypes.POINTER(ctypes.c_float)]
def sum(stream, nelems, a, b, c, d, e):
    status = gpu_matrix_kernels._sum(stream, nelems, a, b, c, d, e)
    cudart.check_cuda_status(status)

gpu_matrix_kernels._slicedInplaceAdd.restype = cudart.cuda_error_t
gpu_matrix_kernels._slicedInplaceAdd.argtypes = [cudart.cuda_stream_t,
                                                 ctypes.c_int,
                                                 ctypes.c_int,
                                                 ctypes.c_float,
                                                 ctypes.POINTER(ctypes.c_float),
                                                 ctypes.POINTER(ctypes.c_int),
                                                 ctypes.POINTER(ctypes.c_float)]
def sliced_inplace_add(stream, nrows, ncols, alpha, dense_matrix, embedding_column_indxs, embedding_matrix):
    status = gpu_matrix_kernels._slicedInplaceAdd(stream, nrows, ncols, alpha, dense_matrix, embedding_column_indxs, embedding_matrix)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._addHadamardProduct.restype = cudart.cuda_error_t
gpu_matrix_kernels._addHadamardProduct.argtypes = [cudart.cuda_stream_t,
                                                   ctypes.c_int,
                                                   ctypes.POINTER(ctypes.c_float),
                                                   ctypes.POINTER(ctypes.c_float),
                                                   ctypes.c_float,
                                                   ctypes.POINTER(ctypes.c_float)]
def add_hadamard_product(stream, nelems, a, b, alpha, c):
    status = gpu_matrix_kernels._addHadamardProduct(stream, nelems, a, b, alpha, c)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._hadamardProduct2.restype = cudart.cuda_error_t
gpu_matrix_kernels._hadamardProduct2.argtypes = [cudart.cuda_stream_t,
                                                 ctypes.c_int,
                                                 ctypes.POINTER(ctypes.c_float),
                                                 ctypes.POINTER(ctypes.c_float),
                                                 ctypes.POINTER(ctypes.c_float)]
def hadamard_product_2(stream, nelems, a, b, c):
    status = gpu_matrix_kernels._hadamardProduct2(stream, nelems, a, b, c)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._hadamardProduct3.restype = cudart.cuda_error_t
gpu_matrix_kernels._hadamardProduct3.argtypes = [cudart.cuda_stream_t,
                                                 ctypes.c_int,
                                                 ctypes.POINTER(ctypes.c_float),
                                                 ctypes.POINTER(ctypes.c_float),
                                                 ctypes.POINTER(ctypes.c_float),
                                                 ctypes.POINTER(ctypes.c_float)]
def hadamard_product_3(stream, nelems, a, b, c, d):
    status = gpu_matrix_kernels._hadamardProduct3(stream, nelems, a, b, c, d)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._sumHprod4.restype = cudart.cuda_error_t
gpu_matrix_kernels._sumHprod4.argtypes = [cudart.cuda_stream_t,
                                          ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float)]
def sum_hprod_4(stream, nelems, a, b, c, d, e):
    status = gpu_matrix_kernels._sumHprod4(stream, nelems, a, b, c, d, e)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._sumHprod5.restype = cudart.cuda_error_t
gpu_matrix_kernels._sumHprod5.argtypes = [cudart.cuda_stream_t,
                                          ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float)]
def sum_hprod_5(stream, nelems, a, b, c, d, e, f):
    status = gpu_matrix_kernels._sumHprod5(stream, nelems, a, b, c, d, e, f)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._sumHprod11.restype = cudart.cuda_error_t
gpu_matrix_kernels._sumHprod11.argtypes = [cudart.cuda_stream_t,
                                           ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float)]
def sum_hprod_11(stream, nelems, a, b, c, d, e, f, g, h, i, j, k, l):
    status = gpu_matrix_kernels._sumHprod11(stream, nelems, a, b, c, d, e, f, g, h, i, j, k, l)
    cudart.check_cuda_status(status)