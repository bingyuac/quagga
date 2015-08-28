import ctypes as ct
import numpy as np
from unittest import TestCase
from quagga.cuda import cudart
from quagga.context import GpuContext
from quagga.cuda.test_events import test_dependencies


def cuda_array_from_list(a):
    a = np.array(a, dtype=np.int32, order='F')
    host_data = a.ctypes.data_as(ct.POINTER(ct.c_int))
    elem_size = ct.sizeof(ct.c_int)
    nbytes = a.size * elem_size
    data = cudart.cuda_malloc(nbytes, ct.c_int)
    cudart.cuda_memcpy(data, host_data, nbytes, 'host_to_device')
    return data


def list_from_cuda_array(a, n, release_memory=True):
    c_int_p = ct.POINTER(ct.c_int)
    host_array = (c_int_p * n)()
    host_ptr = ct.cast(host_array, c_int_p)
    elem_size = ct.sizeof(ct.c_int)
    cudart.cuda_memcpy(host_ptr, a, n * elem_size, 'device_to_host')
    if release_memory:
        cudart.cuda_free(a)
    a = np.ndarray(shape=(n, ), dtype=np.int32, buffer=host_array, order='F')
    return a.tolist()


class TestEvent(TestCase):
    def test_dependencies(self):
        N = 10
        k = 6
        execution_checklist = cuda_array_from_list([0] * (k * N + 1))
        test_results = cuda_array_from_list([0] * (k * N + 1))
        contexts = [GpuContext() for _ in xrange(k)]

        blocking_nodes = list()
        blocking_nodes.append(cuda_array_from_list([]))
        for i in xrange(N):
            blocking_nodes.append(cuda_array_from_list([i*k]))
            blocking_nodes.append(cuda_array_from_list(range(i*k + 1, i*k + 4)))
            blocking_nodes.append(cuda_array_from_list(range(i*k + 4, i*k + 6)))

        for context_id in xrange(5, 6):
            test_dependencies(contexts[context_id].cuda_stream, 0, blocking_nodes[0], 0, execution_checklist, test_results)
            contexts[context_id].block(*contexts[:3])

        for i in xrange(N):
            for context_id in xrange(3):
                test_dependencies(contexts[context_id].cuda_stream, i * k + context_id + 1, blocking_nodes[i*3+1], 1, execution_checklist, test_results)

            for context_id in xrange(3, 5):
                contexts[context_id].wait(*contexts[:3])
                test_dependencies(contexts[context_id].cuda_stream, i * k + context_id + 1, blocking_nodes[i*3+2], 3, execution_checklist, test_results)

            for context_id in xrange(5, 6):
                contexts[context_id].wait(*contexts[3:5])
                test_dependencies(contexts[context_id].cuda_stream, i * k + context_id + 1, blocking_nodes[i*3+3], 2, execution_checklist, test_results)
                contexts[context_id].block(*contexts[:3])

        for context in contexts:
            context.synchronize()

        for nodes in blocking_nodes:
            cudart.cuda_free(nodes)

        test_results = list_from_cuda_array(test_results, k * N + 1)
        execution_checklist = list_from_cuda_array(execution_checklist, k * N + 1)
        self.assertEqual(sum(test_results) + sum(execution_checklist), 2 * (k * N + 1))