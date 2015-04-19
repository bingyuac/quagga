#include "GpuMatrixKernels.cuh"
#include <stdio.h>

namespace kernels {

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


__global__  void hadamardProduct(int nelems,
							     const float* __restrict__ A,
							     const float* __restrict__ B,
							     float alpha,
							     float* __restrict__ C) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		C[i] = A[i] * B[i] + alpha * C[i];
	}
}


__global__  void hadamardProduct(int nelems,
							     const float* __restrict__ A,
							     const float* __restrict__ B,
							     const float* __restrict__ C,
							     float* __restrict__ D) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		D[i] = A[i] * B[i] * C[i];
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
		L[i] = A[i] * B[i] * C[i] + D[i] * E[i] + F[i] * G[i] + H[i] * I[i] + J[i] + K[i];
	}
}


__global__ void sum(int nelems,
						const float* __restrict__ A,
						const float* __restrict__ B,
						const float* __restrict__ C,
						const float* __restrict__ D,
						float* __restrict__ E) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		E[i] = A[i] + B[i] + C[i] + D[i];
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
}
