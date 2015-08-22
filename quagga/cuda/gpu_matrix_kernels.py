import ctypes as ct
from quagga.cuda import cudart


gpu_matrix_kernels = ct.cdll.LoadLibrary('gpu_matrix_kernels.so')


gpu_matrix_kernels._scale.restype = cudart.ct_cuda_error
gpu_matrix_kernels._scale.argtypes = [cudart.ct_cuda_stream,
                                      ct.c_int,
                                      ct.c_float,
                                      ct.POINTER(ct.c_float),
                                      ct.POINTER(ct.c_float)]
def scale(stream, nelems, alpha, data, out_data):
    status = gpu_matrix_kernels._scale(stream, nelems, alpha, data, out_data)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._fill.restype = cudart.ct_cuda_error
gpu_matrix_kernels._fill.argtypes = [cudart.ct_cuda_stream,
                                     ct.c_int,
                                     ct.c_float,
                                     ct.POINTER(ct.c_float)]
def fill(stream, nelems, value, out_data):
    status = gpu_matrix_kernels._fill(stream, nelems, value, out_data)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._add_sum.restype = cudart.ct_cuda_error
gpu_matrix_kernels._add_sum.argtypes = [cudart.ct_cuda_stream,
                                        ct.c_int,
                                        ct.POINTER(ct.POINTER(ct.c_float)),
                                        ct.c_int,
                                        ct.POINTER(ct.c_float)]
def add_sum(stream, nelems, matrices, n, s):
    status = gpu_matrix_kernels._add_sum(stream, nelems, matrices, n, s)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._assign_sum.restype = cudart.ct_cuda_error
gpu_matrix_kernels._assign_sum.argtypes = [cudart.ct_cuda_stream,
                                           ct.c_int,
                                           ct.POINTER(ct.POINTER(ct.c_float)),
                                           ct.c_int,
                                           ct.POINTER(ct.c_float)]
def assign_sum(stream, nelems, matrices, n, s):
    status = gpu_matrix_kernels._assign_sum(stream, nelems, matrices, n, s)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._slicedInplaceAdd.restype = cudart.ct_cuda_error
gpu_matrix_kernels._slicedInplaceAdd.argtypes = [cudart.ct_cuda_stream,
                                                 ct.c_int,
                                                 ct.c_int,
                                                 ct.c_float,
                                                 ct.POINTER(ct.c_float),
                                                 ct.POINTER(ct.c_int),
                                                 ct.POINTER(ct.c_float)]
def sliced_inplace_add(stream, nrows, ncols, alpha, dense_matrix, embedding_column_indxs, embedding_matrix):
    status = gpu_matrix_kernels._slicedInplaceAdd(stream, nrows, ncols, alpha, dense_matrix, embedding_column_indxs, embedding_matrix)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._addHadamardProduct2.restype = cudart.ct_cuda_error
gpu_matrix_kernels._addHadamardProduct2.argtypes = [cudart.ct_cuda_stream,
                                                    ct.c_int,
                                                    ct.POINTER(ct.c_float),
                                                    ct.POINTER(ct.c_float),
                                                    ct.c_float,
                                                    ct.POINTER(ct.c_float)]
def add_hadamard_product_2(stream, nelems, a, b, alpha, c):
    status = gpu_matrix_kernels._addHadamardProduct2(stream, nelems, a, b, alpha, c)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._addHadamardProduct3.restype = cudart.ct_cuda_error
gpu_matrix_kernels._addHadamardProduct3.argtypes = [cudart.ct_cuda_stream,
                                                    ct.c_int,
                                                    ct.POINTER(ct.c_float),
                                                    ct.POINTER(ct.c_float),
                                                    ct.POINTER(ct.c_float),
                                                    ct.c_float,
                                                    ct.POINTER(ct.c_float)]
def add_hadamard_product_3(stream, nelems, a, b, c, alpha, d):
    status = gpu_matrix_kernels._addHadamardProduct3(stream, nelems, a, b, c, alpha, d)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._hadamardProduct2.restype = cudart.ct_cuda_error
gpu_matrix_kernels._hadamardProduct2.argtypes = [cudart.ct_cuda_stream,
                                                 ct.c_int,
                                                 ct.POINTER(ct.c_float),
                                                 ct.POINTER(ct.c_float),
                                                 ct.POINTER(ct.c_float)]
def hadamard_product_2(stream, nelems, a, b, c):
    status = gpu_matrix_kernels._hadamardProduct2(stream, nelems, a, b, c)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._hadamardProduct3.restype = cudart.ct_cuda_error
gpu_matrix_kernels._hadamardProduct3.argtypes = [cudart.ct_cuda_stream,
                                                 ct.c_int,
                                                 ct.POINTER(ct.c_float),
                                                 ct.POINTER(ct.c_float),
                                                 ct.POINTER(ct.c_float),
                                                 ct.POINTER(ct.c_float)]
def hadamard_product_3(stream, nelems, a, b, c, d):
    status = gpu_matrix_kernels._hadamardProduct3(stream, nelems, a, b, c, d)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._sumHprod4.restype = cudart.ct_cuda_error
gpu_matrix_kernels._sumHprod4.argtypes = [cudart.ct_cuda_stream,
                                          ct.c_int,
                                          ct.POINTER(ct.c_float),
                                          ct.POINTER(ct.c_float),
                                          ct.POINTER(ct.c_float),
                                          ct.POINTER(ct.c_float),
                                          ct.POINTER(ct.c_float)]
def sum_hprod_4(stream, nelems, a, b, c, d, e):
    status = gpu_matrix_kernels._sumHprod4(stream, nelems, a, b, c, d, e)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._sumHprod5.restype = cudart.ct_cuda_error
gpu_matrix_kernels._sumHprod5.argtypes = [cudart.ct_cuda_stream,
                                          ct.c_int,
                                          ct.POINTER(ct.c_float),
                                          ct.POINTER(ct.c_float),
                                          ct.POINTER(ct.c_float),
                                          ct.POINTER(ct.c_float),
                                          ct.POINTER(ct.c_float),
                                          ct.POINTER(ct.c_float)]
def sum_hprod_5(stream, nelems, a, b, c, d, e, f):
    status = gpu_matrix_kernels._sumHprod5(stream, nelems, a, b, c, d, e, f)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._sumHprod11.restype = cudart.ct_cuda_error
gpu_matrix_kernels._sumHprod11.argtypes = [cudart.ct_cuda_stream,
                                           ct.c_int,
                                           ct.POINTER(ct.c_float),
                                           ct.POINTER(ct.c_float),
                                           ct.POINTER(ct.c_float),
                                           ct.POINTER(ct.c_float),
                                           ct.POINTER(ct.c_float),
                                           ct.POINTER(ct.c_float),
                                           ct.POINTER(ct.c_float),
                                           ct.POINTER(ct.c_float),
                                           ct.POINTER(ct.c_float),
                                           ct.POINTER(ct.c_float),
                                           ct.POINTER(ct.c_float),
                                           ct.POINTER(ct.c_float)]
def sum_hprod_11(stream, nelems, a, b, c, d, e, f, g, h, i, j, k, l):
    status = gpu_matrix_kernels._sumHprod11(stream, nelems, a, b, c, d, e, f, g, h, i, j, k, l)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._hprodSum.restype = cudart.ct_cuda_error
gpu_matrix_kernels._hprodSum.argtypes = [cudart.ct_cuda_stream,
                                         ct.c_int,
                                         ct.c_int,
                                         ct.POINTER(ct.c_float),
                                         ct.POINTER(ct.c_float),
                                         ct.POINTER(ct.c_float)]
def hprod_sum(stream, nrows, ncols, a, b, c):
    status = gpu_matrix_kernels._hprodSum(stream, nrows, ncols, a, b, c)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._sliceColumns.restype = cudart.ct_cuda_error
gpu_matrix_kernels._sliceColumns.argtypes = [cudart.ct_cuda_stream,
                                             ct.c_int,
                                             ct.c_int,
                                             ct.POINTER(ct.c_int),
                                             ct.POINTER(ct.c_float),
                                             ct.POINTER(ct.c_float)]
def slice_columns(stream, nrows, ncols, embedding_column_indxs, embedding_matrix, dense_matrix):
    status = gpu_matrix_kernels._sliceColumns(stream, nrows, ncols, embedding_column_indxs, embedding_matrix, dense_matrix)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._reverseSliceColumns.restype = cudart.ct_cuda_error
gpu_matrix_kernels._reverseSliceColumns.argtypes = [cudart.ct_cuda_stream,
                                                    ct.c_int,
                                                    ct.c_int,
                                                    ct.POINTER(ct.c_int),
                                                    ct.POINTER(ct.c_float),
                                                    ct.POINTER(ct.c_float)]
def reverse_slice_columns(stream, nrows, ncols, embedding_column_indxs, embedding_matrix, dense_matrix):
    status = gpu_matrix_kernels._reverseSliceColumns(stream, nrows, ncols, embedding_column_indxs, embedding_matrix, dense_matrix)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._verticalStack.restype = cudart.ct_cuda_error
gpu_matrix_kernels._verticalStack.argtypes = [cudart.ct_cuda_stream,
                                              ct.c_int,
                                              ct.POINTER(ct.c_int),
                                              ct.c_int,
                                              ct.POINTER(ct.POINTER(ct.c_float)),
                                              ct.POINTER(ct.c_float)]
def vertical_stack(stream, n, nrows, ncols, matrices, stacked):
    status = gpu_matrix_kernels._verticalStack(stream, n, nrows, ncols, matrices, stacked)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._horizontalStack.restype = cudart.ct_cuda_error
gpu_matrix_kernels._horizontalStack.argtypes = [cudart.ct_cuda_stream,
                                                ct.c_int,
                                                ct.POINTER(ct.c_int),
                                                ct.c_int,
                                                ct.POINTER(ct.POINTER(ct.c_float)),
                                                ct.POINTER(ct.c_float)]
def horizontal_stack(stream, n, ncols, nrows, matrices, stacked):
    status = gpu_matrix_kernels._horizontalStack(stream, n, ncols, nrows, matrices, stacked)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._verticalSplit.restype = cudart.ct_cuda_error
gpu_matrix_kernels._verticalSplit.argtypes = [cudart.ct_cuda_stream,
                                              ct.c_int,
                                              ct.POINTER(ct.c_int),
                                              ct.c_int,
                                              ct.POINTER(ct.POINTER(ct.c_float)),
                                              ct.POINTER(ct.c_float)]
def vertical_split(stream, n, nrows, ncols, matrices, stacked):
    status = gpu_matrix_kernels._verticalSplit(stream, n, nrows, ncols, matrices, stacked)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._horizontalSplit.restype = cudart.ct_cuda_error
gpu_matrix_kernels._horizontalSplit.argtypes = [cudart.ct_cuda_stream,
                                                ct.c_int,
                                                ct.POINTER(ct.c_int),
                                                ct.c_int,
                                                ct.POINTER(ct.POINTER(ct.c_float)),
                                                ct.POINTER(ct.c_float)]
def hotizontal_split(stream, n, ncols, nrows, matrices, stacked):
    status = gpu_matrix_kernels._horizontalSplit(stream, n, ncols, nrows, matrices, stacked)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._verticalSliceSplit.restype = cudart.ct_cuda_error
gpu_matrix_kernels._verticalSliceSplit.argtypes = [cudart.ct_cuda_stream,
                                                   ct.c_int,
                                                   ct.POINTER(ct.c_int),
                                                   ct.c_int,
                                                   ct.c_int,
                                                   ct.POINTER(ct.POINTER(ct.c_float)),
                                                   ct.POINTER(ct.c_float)]
def vertical_slice_split(stream, n, row_slices, nrows, ncols, matrices, stacked):
    status = gpu_matrix_kernels._verticalSliceSplit(stream, n, row_slices, nrows, ncols, matrices, stacked)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._horizontalSliceSplit.restype = cudart.ct_cuda_error
gpu_matrix_kernels._horizontalSliceSplit.argtypes = [cudart.ct_cuda_stream,
                                                     ct.c_int,
                                                     ct.POINTER(ct.c_int),
                                                     ct.c_int,
                                                     ct.POINTER(ct.POINTER(ct.c_float)),
                                                     ct.POINTER(ct.c_float)]
def horizontal_slice_split(stream, n, col_slices, nrows, matrices, stacked):
    status = gpu_matrix_kernels._horizontalSliceSplit(stream, n, col_slices, nrows, matrices, stacked)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._matrixVectorRowAddition.restype = cudart.ct_cuda_error
gpu_matrix_kernels._matrixVectorRowAddition.argtypes = [cudart.ct_cuda_stream,
                                                        ct.c_int,
                                                        ct.c_int,
                                                        ct.POINTER(ct.c_float),
                                                        ct.c_float,
                                                        ct.POINTER(ct.c_float),
                                                        ct.POINTER(ct.c_float)]
def matrix_vector_row_addition(stream, nrows, ncols, matrix, alpha, vector, out):
    status = gpu_matrix_kernels._matrixVectorRowAddition(stream, nrows, ncols, matrix, alpha, vector, out)
    cudart.check_cuda_status(status)


gpu_matrix_kernels._assignSequentialMeanPooling.restype = cudart.ct_cuda_error
gpu_matrix_kernels._assignSequentialMeanPooling.argtypes = [cudart.ct_cuda_stream,
                                                            ct.c_int,
                                                            ct.c_int,
                                                            ct.POINTER(ct.POINTER(ct.c_float)),
                                                            ct.c_int,
                                                            ct.POINTER(ct.c_float)]
def assign_sequential_mean_pooling(stream, nrows, ncols, matrices, n, out):
    status = gpu_matrix_kernels._assignSequentialMeanPooling(stream, nrows, ncols, matrices, n, out)
    cudart.check_cuda_status(status)