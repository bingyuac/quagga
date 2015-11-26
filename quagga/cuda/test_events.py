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