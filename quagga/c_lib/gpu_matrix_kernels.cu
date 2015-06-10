#include <algorithm>
#include <cuda_runtime.h>


#define MAX_NUM_THREADS_PER_BLOCK 512
#define MAX_NUM_BLOCKS_PER_KERNEL 64


__global__  void sliceColumns(int nrows,
							  int ncols,
							  const int* __restrict__ embedding_column_indxs,
							  const float* __restrict__ embedding_matrix,
							  float* __restrict__ dense_matrix) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nelems = nrows * ncols;

	int dense_column_idx;
	int row_idx;
	int embedding_offset;
	for (int i = start_i; i < nelems; i += nthreads) {
		dense_column_idx = i / nrows;
		row_idx = i % nrows;
		embedding_offset = embedding_column_indxs[dense_column_idx] * nrows + row_idx;
		dense_matrix[i] = embedding_matrix[embedding_offset];
	}
}


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


__global__ void fill(int nelems, float val, float* __restrict__ A) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		A[i] = val;
	}
}

__global__ void hprodSum(int nelems,
						 int nrows,
						 const float* __restrict__ A,
						 const float* __restrict__ B,
						 float* __restrict__ C) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		atomicAdd(C + i % nrows, A[i] * B[i]);
	}
}


__global__ void sumHprod(int nelems,
						 const float* __restrict__ A,
						 const float* __restrict__ B,
						 const float* __restrict__ C,
						 const float* __restrict__ D,
						 float* __restrict__ E) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		E[i] = A[i] * B[i] + C[i] * D[i];
	}
}


__global__ void sumHprod(int nelems,
						 const float* __restrict__ A,
						 const float* __restrict__ B,
						 const float* __restrict__ C,
						 const float* __restrict__ D,
						 const float* __restrict__ E,
						 float* __restrict__ F) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		F[i] = A[i] * B[i] * C[i] + D[i] * E[i];
	}
}


__global__ void sumHprod(int nelems,
						 const float* __restrict__ A,
						 const float* __restrict__ B,
						 const float* __restrict__ C,
						 const float* __restrict__ D,
						 const float* __restrict__ E,
						 const float* __restrict__ F,
						 const float* __restrict__ G,
						 const float* __restrict__ H,
						 const float* __restrict__ I,
						 const float* __restrict__ J,
						 const float* __restrict__ K,
						 float* __restrict__ L) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		L[i] = A[i] * B[i] * C[i] + D[i] * E[i] + F[i] * G[i] + H[i] * I[i] + J[i] * K[i];
	}
}


__global__  void hadamardProduct(int nelems,
							     const float* __restrict__ a,
							     const float* __restrict__ b,
							     float* __restrict__ c) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		c[i] = a[i] * b[i];
	}
}


__global__  void hadamardProduct(int nelems,
							     const float* __restrict__ a,
							     const float* __restrict__ b,
							     const float* __restrict__ c,
							     float* __restrict__ d) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		d[i] = a[i] * b[i] * c[i];
	}
}


__global__  void addHadamardProduct(int nelems,
							        const float* __restrict__ a,
							        const float* __restrict__ b,
							        float alpha,
							        float* __restrict__ c) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		c[i] = a[i] * b[i] + alpha * c[i];
	}
}


__global__  void slicedInplaceAdd(int nrows,
							      int ncols,
							      float alpha,
							      const float* __restrict__ dense_matrix,
							      const int* __restrict__ embedding_column_indxs,
							      float* __restrict__ embedding_matrix) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nelems = nrows * ncols;

	int dense_column_idx;
	int row_idx;
	int embedding_offset;
	for (int i = start_i; i < nelems; i += nthreads) {
		dense_column_idx = i / nrows;
		row_idx = i % nrows;
		embedding_offset = embedding_column_indxs[dense_column_idx] * nrows + row_idx;
		atomicAdd(embedding_matrix + embedding_offset, alpha * dense_matrix[i]);
	}
}


__global__ void sum(int nelems,
					const float* __restrict__ a,
					const float* __restrict__ b,
					const float* __restrict__ c,
					const float* __restrict__ d,
					float* __restrict__ e) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		e[i] = a[i] + b[i] + c[i] + d[i];
	}
}


__global__ void sigmoid(int nelems,
						const float* __restrict__ data,
						float* __restrict__ sigmoid_data) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		sigmoid_data[i] = 1.0f / (1.0f + expf(-data[i]));
	}
}


__global__ void sigmoid(int nelems,
						const float* __restrict__ data,
						float* __restrict__ sigmoid_data,
						float* __restrict__ derivative) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		sigmoid_data[i] = 1.0f / (1.0f + expf(-data[i]));
		derivative[i] = sigmoid_data[i] * (1.0f - sigmoid_data[i]);
	}
}


__global__ void tanh(int nelems,
					 const float* __restrict__ data,
					 float* __restrict__ tanh_data) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		tanh_data[i] = tanhf(data[i]);
	}
}


__global__ void tanh(int nelems,
					 const float* __restrict__ data,
					 float* __restrict__ tanh_data,
					 float* __restrict__ derivative) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		tanh_data[i] = tanhf(data[i]);
		derivative[i] = 1.0f - tanh_data[i] * tanh_data[i];
	}
}


__global__ void scale(int nelems,
					  const float* __restrict__ data,
					  float alpha,
					  float* __restrict__ out_data) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		out_data[i] = alpha * data[i];
	}
}


extern "C" {
	cudaError_t _sliceColumns(cudaStream_t stream,
							  int nrows,
							  int ncols,
							  const int* __restrict__ embedding_column_indxs,
							  const float* __restrict__ embedding_matrix,
							  float* __restrict__ dense_matrix) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
		sliceColumns<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, embedding_column_indxs, embedding_matrix, dense_matrix);
		return cudaGetLastError();
	}


	cudaError_t _testDependencies(cudaStream_t stream, int node_id, int* blocking_nodes, int blocking_nodes_num, int *execution_checklist, int* test_results) {
		testDependencies<<<1, 1, 0, stream>>>(node_id, blocking_nodes, blocking_nodes_num, execution_checklist, test_results);
		return cudaGetLastError();
	}


	cudaError_t _hprodSum(cudaStream_t stream,
                          int nrows,
                          int ncols,
						  const float* __restrict__ a,
						  const float* __restrict__ b,
						  float* __restrict__ c) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
		fill<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, 0.0, c);
		int nelems = nrows * ncols;
		num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        hprodSum<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, nrows, a, b, c);
        return cudaGetLastError();
	}


    cudaError_t _sumHprod4(cudaStream_t stream,
                           int nelems,
						   const float* __restrict__ a,
						   const float* __restrict__ b,
						   const float* __restrict__ c,
						   const float* __restrict__ d,
						   float* __restrict__ e) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sumHprod<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, c, d, e);
        return cudaGetLastError();
    }


    cudaError_t _sumHprod5(cudaStream_t stream,
                           int nelems,
						   const float* __restrict__ a,
						   const float* __restrict__ b,
						   const float* __restrict__ c,
						   const float* __restrict__ d,
						   const float* __restrict__ e,
						   float* __restrict__ f) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sumHprod<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, c, d, e, f);
        return cudaGetLastError();
    }


    cudaError_t _sumHprod11(cudaStream_t stream,
                            int nelems,
						    const float* __restrict__ a,
						    const float* __restrict__ b,
						    const float* __restrict__ c,
						    const float* __restrict__ d,
						    const float* __restrict__ e,
						    const float* __restrict__ f,
						    const float* __restrict__ g,
						    const float* __restrict__ h,
						    const float* __restrict__ i,
						    const float* __restrict__ j,
						    const float* __restrict__ k,
						    float* __restrict__ l) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sumHprod<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, c, d, e, f, g, h, i, j, k, l);
        return cudaGetLastError();
    }


    cudaError_t _hadamardProduct2(cudaStream_t stream,
                                  int nelems,
						     	  const float* __restrict__ a,
							      const float* __restrict__ b,
							      float* __restrict__ c) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        hadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, c);
        return cudaGetLastError();
    }


    cudaError_t _hadamardProduct3(cudaStream_t stream,
                                  int nelems,
						     	  const float* __restrict__ a,
							      const float* __restrict__ b,
							      const float* __restrict__ c,
							      float* __restrict__ d) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        hadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, c, d);
        return cudaGetLastError();
    }


    cudaError_t _addHadamardProduct(cudaStream_t stream,
                                    int nelems,
				 			        const float* __restrict__ a,
							        const float* __restrict__ b,
							        float alpha,
							        float* __restrict__ c) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addHadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, alpha, c);
        return cudaGetLastError();
    }


    cudaError_t _slicedInplaceAdd(cudaStream_t stream,
                                  int nrows,
							      int ncols,
							      float alpha,
							      const float* __restrict__ dense_matrix,
							      const int* __restrict__ embedding_column_indxs,
							      float* __restrict__ embedding_matrix) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        slicedInplaceAdd<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, alpha, dense_matrix, embedding_column_indxs, embedding_matrix);
        return cudaGetLastError();
	}

    cudaError_t _sum(cudaStream_t stream,
                     int nelems,
					 const float* __restrict__ a,
					 const float* __restrict__ b,
					 const float* __restrict__ c,
					 const float* __restrict__ d,
					 float* __restrict__ e) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sum<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, c, d, e);
        return cudaGetLastError();
	}


    cudaError_t _sigmoid(cudaStream_t stream,
                         int nelems,
			             const float* __restrict__ data,
			             float* __restrict__ sigmoid_data) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sigmoid<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, data, sigmoid_data);
        return cudaGetLastError();
	}


	cudaError_t _sigmoid_der(cudaStream_t stream,
                             int nelems,
			                 const float* __restrict__ data,
			                 float* __restrict__ sigmoid_data,
			                 float* __restrict__ derivative) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sigmoid<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, data, sigmoid_data, derivative);
        return cudaGetLastError();
	}


    cudaError_t _tanh(cudaStream_t stream,
                      int nelems,
			          const float* __restrict__ data,
			          float* __restrict__ tanh_data) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        tanh<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, data, tanh_data);
        return cudaGetLastError();
	}


	cudaError_t _tanh_der(cudaStream_t stream,
                          int nelems,
			              const float* __restrict__ data,
			              float* __restrict__ tanh_data,
			              float* __restrict__ derivative) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        tanh<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, data, tanh_data, derivative);
        return cudaGetLastError();
	}


	cudaError_t _scale(cudaStream_t stream,
	                   int nelems,
	                   float alpha,
                       const float* __restrict__ data,
                       float* __restrict__ out_data) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        scale<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, data, alpha, out_data);
        return cudaGetLastError();
    }
}
