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


__global__  void reverseSliceColumns(int nrows,
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
	int dense_offset;
	for (int i = start_i; i < nelems; i += nthreads) {
		dense_column_idx = i / nrows;
		row_idx = i % nrows;
		embedding_offset = embedding_column_indxs[dense_column_idx] * nrows + row_idx;
		dense_offset = nrows * (ncols - 1 - dense_column_idx) + row_idx;
		dense_matrix[dense_offset] = embedding_matrix[embedding_offset];
	}
}


__global__  void sliceRowsBatch(const int* embd_rows_indxs,
								int nrows,
								int ncols,
							    const float* __restrict__ embd_matrix,
							    int embd_nrows,
							    int embd_ncols,
							    float* __restrict__ dense_matrices[]) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nelems = nrows * embd_ncols;
	const int total_nelems = nelems * ncols;

	int k, dense_offset, embd_row_idx, embd_col_idx, embd_offset;
	for (int i = start_i; i < total_nelems; i += nthreads) {
		k = i / nelems;
		dense_offset = i % nelems;
		embd_row_idx = embd_rows_indxs[k * nrows + i % nrows];
		embd_col_idx = dense_offset / nrows;
		embd_offset = embd_col_idx * embd_nrows + embd_row_idx;
		dense_matrices[k][dense_offset] = embd_matrix[embd_offset];
	}
}


__global__  void reverseSliceRowsBatch(const int* embd_rows_indxs,
									   int nrows,
									   int ncols,
							    	   const float* __restrict__ embd_matrix,
							    	   int embd_nrows,
							    	   int embd_ncols,
							    	   float* __restrict__ dense_matrices[]) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nelems = nrows * embd_ncols;
	const int total_nelems = nelems * ncols;

	int k, dense_offset, embd_row_idx, embd_col_idx, embd_offset;
	for (int i = start_i; i < total_nelems; i += nthreads) {
		k = i / nelems;
		dense_offset = i % nelems;
		embd_row_idx = embd_rows_indxs[(ncols - 1 - k) * nrows + i % nrows];
		embd_col_idx = dense_offset / nrows;
		embd_offset = embd_col_idx * embd_nrows + embd_row_idx;
		dense_matrices[k][dense_offset] = embd_matrix[embd_offset];
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


__global__  void addHadamardProduct(int nelems,
							        const float* __restrict__ a,
							        const float* __restrict__ b,
							        const float* __restrict__ c,
							        float alpha,
							        float* __restrict__ d) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		d[i] = a[i] * b[i] * c[i] + alpha * d[i];
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


__global__ void assignSum(int nelems,
						  const float* matrices[],
						  int n,
						  float* __restrict__ s) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		s[i] = 0.0;
		for (int k = 0; k < n; k++) {
			s[i] += matrices[k][i];
		}
	}
}


__global__ void addSum(int nelems,
					   const float* matrices[],
					   int n,
					   float* __restrict__ s) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		for (int k = 0; k < n; k++) {
			s[i] += matrices[k][i];
		}
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


__global__ void fill(int nelems,
					 float value,
					 float* __restrict__ out_data) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		out_data[i] = value;
	}
}


__global__ void matrixVectorRowAddition(int nrows,
							      		int ncols,
							      		const float* matrix,
							      		float alpha,
							      		const float* vector,
							      		float* out) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nelems = nrows * ncols;

	for (int i = start_i; i < nelems; i += nthreads) {
		out[i] = matrix[i] + alpha * vector[i / nrows];
	}
}


__global__ void assignScaledAddition(int nelems,
							      	 float alpha,
							      	 const float* a,
							      	 const float* b,
							      	 float* out) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		out[i] = alpha * (a[i] + b[i]);
	}
}


__global__ void assignScaledSubtraction(int nelems,
							      	    float alpha,
							      	 	const float* a,
							      	 	const float* b,
							      	 	float* out) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		out[i] = alpha * (a[i] - b[i]);
	}
}


__global__ void assignSequentialMeanPooling(int nrows,
                     						int ncols,
                     						const float* matrices[],
                     						int n,
                     						float* __restrict__ out) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nelems = nrows * ncols;
	const int total_nelems = nelems * n;

	int k, m;
	for (int i = start_i; i < total_nelems; i += nthreads) {
		k = i / nelems;
		m = i % nelems;
		atomicAdd(out + m, matrices[k][m] / n);
	}
}


__global__ void sequentiallyTile(int nelems,
                     			 const float* __restrict__ a,
                     			 float* matrices[],
                     			 int n) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int total_nelems = nelems * n;

	int k, m;
	for (int i = start_i; i < total_nelems; i += nthreads) {
		k = i / nelems;
		m = i % nelems;
		matrices[k][m] = a[m];
	}
}


__global__ void dropout(int nelems,
						float dropout_prob,
						const float* __restrict__ data,
						const float* __restrict__ uniform_data,
						float* __restrict__ out) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		out[i] = data[i] * (uniform_data[i] > dropout_prob);
	}
}


__global__ void maskZeros(int nelems,
					 	  const float* __restrict__ a,
					 	  const float* __restrict__ b,
					 	  float* __restrict__ out) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		out[i] = a[i] * (b[i] != 0.0);
	}
}


extern "C" {
	cudaError_t _maskZeros(cudaStream_t stream,
                           int nelems,
			               const float* __restrict__ a,
			               const float* __restrict__ b,
			               float* __restrict__ out) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        maskZeros<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, out);
        return cudaGetLastError();
	}


	cudaError_t _dropout(cudaStream_t stream,
                         int nelems,
                         float dropout_prob,
			             const float* __restrict__ data,
						 const float* __restrict__ uniform_data,
						 float* __restrict__ out) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        dropout<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, dropout_prob, data, uniform_data, out);
        return cudaGetLastError();
	}


	cudaError_t _horizontalSliceSplit(cudaStream_t stream,
						   			  int n,
						   			  int* col_slices,
						   			  int nrows,
						   			  float** matrices,
						   			  float* stacked) {
		size_t float_size = sizeof(float);
		int nelems;
		int offset = 0;

		for (int i = 0; i < n; i++) {
			nelems = (col_slices[i*2+1] - col_slices[i*2]) * nrows;
			offset = col_slices[i*2] * nrows;
			cudaMemcpyAsync(matrices[i], stacked + offset, float_size * nelems, cudaMemcpyDeviceToDevice, stream);
		}
		return cudaGetLastError();
	}

	cudaError_t _horizontalSplit(cudaStream_t stream,
							     int n,
							  	 int* ncols,
							   	 int nrows,
							   	 float** matrices,
							   	 float* stacked) {
		size_t float_size = sizeof(float);
		int nelems;
		int offset = 0;

		for (int i = 0; i < n; i++) {
			nelems = ncols[i] * nrows;
			cudaMemcpyAsync(matrices[i], stacked + offset, float_size * nelems, cudaMemcpyDeviceToDevice, stream);
			offset += nelems;
		}
		return cudaGetLastError();
	}

	cudaError_t _horizontalStack(cudaStream_t stream,
							   	 int n,
							   	 int* ncols,
							   	 int nrows,
							   	 float** matrices,
							   	 float* stacked) {
		size_t float_size = sizeof(float);
		int nelems;
		int offset = 0;

		for (int i = 0; i < n; i++) {
			nelems = ncols[i] * nrows;
			cudaMemcpyAsync(stacked + offset, matrices[i], float_size * nelems, cudaMemcpyDeviceToDevice, stream);
			offset += nelems;
		}
		return cudaGetLastError();
	}


	cudaError_t _verticalSliceSplit(cudaStream_t stream,
						   			int n,
						   			int* row_slices,
						   			int nrows,
						   			int ncols,
						   			float** matrices,
						   			float* stacked) {
		size_t float_size = sizeof(float);
		float* column_address;
		int offset = 0;
		int k;

		for (int i = 0; i < ncols; i++) {
			for (int j = 0; j < n; j++) {
				k = row_slices[j*2+1] - row_slices[j*2];
				column_address = matrices[j] + k * i;
				offset = nrows * i + row_slices[j*2];
				cudaMemcpyAsync(column_address, stacked + offset, float_size * k, cudaMemcpyDeviceToDevice, stream);
			}
		}

		return cudaGetLastError();
	}

	cudaError_t _verticalSplit(cudaStream_t stream,
							   int n,
							   int* nrows,
							   int ncols,
							   float** matrices,
							   float* stacked) {
		size_t float_size = sizeof(float);
		float* column_address;
		int offset = 0;

		for (int i = 0; i < ncols; i++) {
			for (int j = 0; j < n; j++) {
				column_address = matrices[j] + nrows[j] * i;
				cudaMemcpyAsync(column_address, stacked + offset, float_size * nrows[j], cudaMemcpyDeviceToDevice, stream);
				offset += nrows[j];
			}
		}
		return cudaGetLastError();
	}


	cudaError_t _verticalStack(cudaStream_t stream,
							   int n,
							   int* nrows,
							   int ncols,
							   float** matrices,
							   float* stacked) {
		size_t float_size = sizeof(float);
		float* column_address;
		int offset = 0;

		for (int i = 0; i < ncols; i++) {
			for (int j = 0; j < n; j++) {
				column_address = matrices[j] + nrows[j] * i;
				cudaMemcpyAsync(stacked + offset, column_address, float_size * nrows[j], cudaMemcpyDeviceToDevice, stream);
				offset += nrows[j];
			}
		}

		return cudaGetLastError();
	}


	cudaError_t _sliceColumns(cudaStream_t stream,
							  int nrows,
							  int ncols,
							  const int* __restrict__ embedding_column_indxs,
							  const float* __restrict__ embedding_matrix,
							  float* __restrict__ dense_matrix) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols  - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
		sliceColumns<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, embedding_column_indxs, embedding_matrix, dense_matrix);
		return cudaGetLastError();
	}


	cudaError_t _reverseSliceColumns(cudaStream_t stream,
							  		 int nrows,
							  		 int ncols,
							  		 const int* __restrict__ embedding_column_indxs,
							  		 const float* __restrict__ embedding_matrix,
							  		 float* __restrict__ dense_matrix) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols  - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
		reverseSliceColumns<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, embedding_column_indxs, embedding_matrix, dense_matrix);
		return cudaGetLastError();
	}


	cudaError_t _sliceRowsBatch(cudaStream_t stream,
								const int* embd_rows_indxs,
								int nrows,
								int ncols,
							    const float* __restrict__ embd_matrix,
							    int embd_nrows,
							    int embd_ncols,
							    float* __restrict__ dense_matrices[]) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * embd_ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sliceRowsBatch<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(embd_rows_indxs, nrows, ncols, embd_matrix, embd_nrows, embd_ncols, dense_matrices);
        return cudaGetLastError();
	}


	cudaError_t _reverseSliceRowsBatch(cudaStream_t stream,
									   const int* embd_rows_indxs,
									   int nrows,
									   int ncols,
								       const float* __restrict__ embd_matrix,
								       int embd_nrows,
								       int embd_ncols,
								       float* __restrict__ dense_matrices[]) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * embd_ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        reverseSliceRowsBatch<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(embd_rows_indxs, nrows, ncols, embd_matrix, embd_nrows, embd_ncols, dense_matrices);
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


    cudaError_t _addHadamardProduct2(cudaStream_t stream,
                                   	 int nelems,
				 			       	 const float* __restrict__ a,
							       	 const float* __restrict__ b,
							       	 float alpha,
							       	 float* __restrict__ c) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addHadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, alpha, c);
        return cudaGetLastError();
    }


    cudaError_t _addHadamardProduct3(cudaStream_t stream,
                                     int nelems,
				 			         const float* __restrict__ a,
							         const float* __restrict__ b,
							         const float* __restrict__ c,
							         float alpha,
							         float* __restrict__ d) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addHadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, b, c, alpha, d);
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


    cudaError_t _add_sum(cudaStream_t stream,
                     int nelems,
                     const float* matrices[],
                     int n,
                     float* __restrict__ s) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        addSum<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, matrices, n, s);
        return cudaGetLastError();
	}


	cudaError_t _assign_sum(cudaStream_t stream,
                     		int nelems,
                     		const float* matrices[],
                     		int n,
                     		float* __restrict__ s) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        assignSum<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, matrices, n, s);
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


    cudaError_t _fill(cudaStream_t stream,
	                  int nelems,
	                  float value,
                      float* __restrict__ out_data) {
        int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        fill<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, value, out_data);
        return cudaGetLastError();
    }


    cudaError_t _matrixVectorRowAddition(cudaStream_t stream,
    									 int nrows,
							      		 int ncols,
							      		 const float* matrix,
							      		 float alpha,
							      		 const float* vector,
							      		 float* out) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        matrixVectorRowAddition<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, matrix, alpha, vector, out);
        return cudaGetLastError();
    }


    cudaError_t _assignSequentialMeanPooling(cudaStream_t stream,
                      						 int nrows,
                      						 int ncols,
                     						 const float* matrices[],
                     						 int n,
                     						 float* __restrict__ out) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        assignSequentialMeanPooling<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, matrices, n, out);
        return cudaGetLastError();
	}


	cudaError_t _sequentiallyTile(cudaStream_t stream,
                      		   	  int nelems,
                     			  const float* __restrict__ a,
                     			  float* matrices[],
                     			  int n) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sequentiallyTile<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, a, matrices, n);
        return cudaGetLastError();
	}


	cudaError_t _assignScaledAddition(cudaStream_t stream,
                      		   	  	  int nelems,
							      	  float alpha,
							      	  const float* a,
							      	  const float* b,
							      	  float* out) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        assignScaledAddition<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, alpha, a, b, out);
        return cudaGetLastError();
	}

	cudaError_t _assignScaledSubtraction(cudaStream_t stream,
                      		   	  	  	 int nelems,
							      	  	 float alpha,
							      	  	 const float* a,
							      	  	 const float* b,
							      	  	 float* out) {
		int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        assignScaledSubtraction<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, alpha, a, b, out);
        return cudaGetLastError();
	}
}