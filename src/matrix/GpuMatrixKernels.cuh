#ifndef MATRIXKERNELS_CUH_
#define MATRIXKERNELS_CUH_

#define MAX_NUM_THREADS_PER_BLOCK 1024
#define MAX_NUM_BLOCKS_PER_KERNEL 16

namespace kernels {

__global__  void sliceColumns(int nrows,
									int ncols,
									const int* __restrict__ embedding_column_indxs,
									const float* __restrict__ embedding_matrix,
									float* __restrict__ dense_matrix);
__global__  void slicedInplaceAdd(int nrows,
										int ncols,
										float alpha,
										const float* __restrict__ dense_matrix,
									    const int* __restrict__ embedding_column_indxs,
									    float* __restrict__ embedding_matrix);
__global__  void hadamardProduct(int nelems,
							     const float* __restrict__ A,
							     const float* __restrict__ B,
							     float alpha,
							     float* __restrict__ C);
__global__  void hadamardProduct(int nelems,
							     const float* __restrict__ A,
							     const float* __restrict__ B,
							     const float* __restrict__ C,
							     float* __restrict__ D);
__global__ void tanh(int nelems,
					 const float* __restrict__ data,
					 float* __restrict__ tanh_data);
__global__ void tanh(int nelems,
					 const float* __restrict__ data,
					 float* __restrict__ tanh_data,
					 float* __restrict__ derivative);
__global__ void sigmoid(int nelems,
						const float* __restrict__ data,
						float* __restrict__ sigmoid_data);
__global__ void sigmoid(int nelems,
						const float* __restrict__ data,
						float* __restrict__ sigmoid_data,
						float* __restrict__ derivative);
__global__ void sumHprod(int nelems,
						const float* __restrict__ A,
						const float* __restrict__ B,
						const float* __restrict__ C,
						const float* __restrict__ D,
						float* __restrict__ E);
__global__ void sumHprod(int nelems,
						const float* __restrict__ A,
						const float* __restrict__ B,
						const float* __restrict__ C,
						const float* __restrict__ D,
						const float* __restrict__ E,
						float* __restrict__ F);
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
						float* __restrict__ L);
__global__ void sum(int nelems,
					const float* __restrict__ A,
					const float* __restrict__ B,
					const float* __restrict__ C,
					const float* __restrict__ D,
					float* __restrict__ E);
__global__ void scale(int nelems,
					  const float* __restrict__ data,
					  float alpha,
					  float* __restrict__ out_data);
}

#endif
