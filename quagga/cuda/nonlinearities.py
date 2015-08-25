import ctypes as ct
from quagga.cuda import cudart


nonlinearities = ct.cdll.LoadLibrary('nonlinearities.so')


nonlinearities._relu.restype = cudart.ct_cuda_error
nonlinearities._relu.argtypes = [cudart.ct_cuda_stream,
                                 ct.c_int,
                                 ct.POINTER(ct.c_float),
                                 ct.POINTER(ct.c_float)]
def relu(stream, nelems, data, relu_data):
    status = nonlinearities._relu(stream, nelems, data, relu_data)
    cudart.check_cuda_status(status)


nonlinearities._reluDer.restype = cudart.ct_cuda_error
nonlinearities._reluDer.argtypes = [cudart.ct_cuda_stream,
                                    ct.c_int,
                                    ct.POINTER(ct.c_float),
                                    ct.POINTER(ct.c_float),
                                    ct.POINTER(ct.c_float)]
def relu_der(stream, nelems, data, relu_data, derivative):
    status = nonlinearities._reluDer(stream, nelems, data, relu_data, derivative)
    cudart.check_cuda_status(status)


nonlinearities._sigmoid.restype = cudart.ct_cuda_error
nonlinearities._sigmoid.argtypes = [cudart.ct_cuda_stream,
                                    ct.c_int,
                                    ct.POINTER(ct.c_float),
                                    ct.POINTER(ct.c_float)]
def sigmoid(stream, nelems, data, sigmoid_data):
    status = nonlinearities._sigmoid(stream, nelems, data, sigmoid_data)
    cudart.check_cuda_status(status)


nonlinearities._sigmoidDer.restype = cudart.ct_cuda_error
nonlinearities._sigmoidDer.argtypes = [cudart.ct_cuda_stream,
                                       ct.c_int,
                                       ct.POINTER(ct.c_float),
                                       ct.POINTER(ct.c_float),
                                       ct.POINTER(ct.c_float)]
def sigmoid_der(stream, nelems, data, sigmoid_data, derivative):
    status = nonlinearities._sigmoidDer(stream, nelems, data, sigmoid_data, derivative)
    cudart.check_cuda_status(status)


nonlinearities._tanh.restype = cudart.ct_cuda_error
nonlinearities._tanh.argtypes = [cudart.ct_cuda_stream,
                                 ct.c_int,
                                 ct.POINTER(ct.c_float),
                                 ct.POINTER(ct.c_float)]
def tanh(stream, nelems, data, tanh_data):
    status = nonlinearities._tanh(stream, nelems, data, tanh_data)
    cudart.check_cuda_status(status)


nonlinearities._tanhDer.restype = cudart.ct_cuda_error
nonlinearities._tanhDer.argtypes = [cudart.ct_cuda_stream,
                                    ct.c_int,
                                    ct.POINTER(ct.c_float),
                                    ct.POINTER(ct.c_float),
                                    ct.POINTER(ct.c_float)]
def tanh_der(stream, nelems, data, tanh_data, derivative):
    status = nonlinearities._tanhDer(stream, nelems, data, tanh_data, derivative)
    cudart.check_cuda_status(status)


nonlinearities._tanhSigm.restype = cudart.ct_cuda_error
nonlinearities._tanhSigm.argtypes = [cudart.ct_cuda_stream,
                                     ct.c_int,
                                     ct.c_int,
                                     ct.c_int,
                                     ct.POINTER(ct.c_float),
                                     ct.POINTER(ct.c_float)]
def tanh_sigm(stream, axis, nrows, ncols, data, tanh_sigm_data):
    status = nonlinearities._tanhSigm(stream, axis, nrows, ncols, data, tanh_sigm_data)
    cudart.check_cuda_status(status)


nonlinearities._tanhSigmDer.restype = cudart.ct_cuda_error
nonlinearities._tanhSigmDer.argtypes = [cudart.ct_cuda_stream,
                                        ct.c_int,
                                        ct.c_int,
                                        ct.c_int,
                                        ct.POINTER(ct.c_float),
                                        ct.POINTER(ct.c_float),
                                        ct.POINTER(ct.c_float)]
def tanh_sigm_der(stream, axis, nrows, ncols, data, tanh_sigm_data, derivatve):
    status = nonlinearities._tanhSigmDer(stream, axis, nrows, ncols, data, tanh_sigm_data, derivatve)
    cudart.check_cuda_status(status)