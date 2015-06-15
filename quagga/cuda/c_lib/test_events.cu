#include <cuda_runtime.h>


__global__ void testDependencies(int node_id, int* blocking_nodes, int blocking_nodes_num, int *execution_checklist, int* test_results) {
	test_results[node_id] = 1;
	for (int i = 0; i < blocking_nodes_num; i++) {
		int bloking_node_id = blocking_nodes[i];
		if (!execution_checklist[bloking_node_id]) {
			test_results[node_id] = 0;
			break;
		}
	}

	clock_t start_clock = clock64();
    clock_t clock_offset = 0;
    while (clock_offset < 4000000000L) {
        clock_offset = clock64() - start_clock;
    }

    execution_checklist[node_id] = 1;
}


extern "C" {
    cudaError_t _testDependencies(cudaStream_t stream, int node_id, int* blocking_nodes, int blocking_nodes_num, int *execution_checklist, int* test_results) {
		testDependencies<<<1, 1, 0, stream>>>(node_id, blocking_nodes, blocking_nodes_num, execution_checklist, test_results);
		return cudaGetLastError();
    }
}