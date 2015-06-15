import ctypes as ct
from quagga.cuda import cudart


test_events = ct.cdll.LoadLibrary('test_events.so')
test_events._testDependencies.restype = cudart.ct_cuda_error
test_events._testDependencies.argtypes = [cudart.ct_cuda_stream,
                                          ct.c_int,
                                          ct.POINTER(ct.c_int),
                                          ct.c_int,
                                          ct.POINTER(ct.c_int),
                                          ct.POINTER(ct.c_int)]
def test_dependencies(cuda_stream, node_id, blocking_nodes, blocking_nodes_num, execution_checklist, test_results):
    status = test_events._testDependencies(cuda_stream, node_id, blocking_nodes, blocking_nodes_num, execution_checklist, test_results)
    cudart.check_cuda_status(status)