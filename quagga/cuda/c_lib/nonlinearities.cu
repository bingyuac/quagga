#include <algorithm>
#include <cuda_runtime.h>


#define MAX_NUM_THREADS_PER_BLOCK 512
#define MAX_NUM_BLOCKS_PER_KERNEL 128


__global__ void sigmoid(int nelems,
						const float* __restrict__ data,
						float* __restrict__ sigmoidData) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		sigmoidData[i] = 1.0f / (1.0f + expf(-data[i]));
	}
}


__global__ void sigmoid(int nelems,
						const float* __restrict__ data,
						float* __restrict__ sigmoidData,
						float* __restrict__ derivative) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		sigmoidData[i] = 1.0f / (1.0f + expf(-data[i]));
		derivative[i] = sigmoidData[i] * (1.0f - sigmoidData[i]);
	}
}


__global__ void tanh(int nelems,
					 const float* __restrict__ data,
					 float* __restrict__ tanhData) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		tanhData[i] = tanhf(data[i]);
	}
}


__global__ void tanh(int nelems,
					 const float* __restrict__ data,
					 float* __restrict__ tanhData,
					 float* __restrict__ derivative) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		tanhData[i] = tanhf(data[i]);
		derivative[i] = 1.0f - tanhData[i] * tanhData[i];
	}
}


__global__ void tanhSigmRow(int nrows,
						 	int ncols,
						 	const float* __restrict__ data,
						 	float* __restrict__ tanhSigmData) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nelems = nrows * ncols;
	const int margin  = nrows / 4;

	for (int i = start_i; i < nelems; i += nthreads) {
		if (i % nrows < margin) {
			tanhSigmData[i] = tanhf(data[i]);
		} else {
			tanhSigmData[i] = 1.0f / (1.0f + expf(-data[i]));
		}
	}
}


__global__ void tanhSigmRow(int nrows,
						 	int ncols,
						 	const float* __restrict__ data,
						 	float* __restrict__ tanhSigmData,
						 	float* __restrict__ derivative) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nelems = nrows * ncols;
	const int margin  = nrows / 4;

	for (int i = start_i; i < nelems; i += nthreads) {
		if (i % nrows < margin) {
			tanhSigmData[i] = tanhf(data[i]);
			derivative[i] = 1.0f - tanhSigmData[i] * tanhSigmData[i];
		} else {
			tanhSigmData[i] = 1.0f / (1.0f + expf(-data[i]));
			derivative[i] = tanhSigmData[i] * (1.0f - tanhSigmData[i]);
		}
	}
}


__global__ void tanhSigmColumn(int nrows,
						 	   int ncols,
						 	   const float* __restrict__ data,
						 	   float* __restrict__ tanhSigmData) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nelems  = ncols * nrows;
	const int margin  = ncols / 4 * nrows;
	int i;

	for (i = start_i; i < margin; i += nthreads) {
		tanhSigmData[i] = tanhf(data[i]);
	}
	for (; i < nelems; i += nthreads) {
		tanhSigmData[i] = 1.0f / (1.0f + expf(-data[i]));
	}
}


__global__ void tanhSigmColumn(int nrows,
						 	   int ncols,
						 	   const float* __restrict__ data,
						 	   float* __restrict__ tanhSigmData,
						 	   float* __restrict__ derivative) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nelems  = ncols * nrows;
	const int margin  = ncols / 4 * nrows;
	int i;

	for (i = start_i; i < margin; i += nthreads) {
		tanhSigmData[i] = tanhf(data[i]);
		derivative[i] = 1.0f - tanhSigmData[i] * tanhSigmData[i];
	}
	for (; i < nelems; i += nthreads) {
		tanhSigmData[i] = 1.0f / (1.0f + expf(-data[i]));
		derivative[i] = tanhSigmData[i] * (1.0f - tanhSigmData[i]);
	}
}


__global__ void relu(int nelems,
				 	 const float* __restrict__ data,
					 float* __restrict__ reluData) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		reluData[i] = fmaxf(0.0f, data[i]);
	}
}


__global__ void relu(int nelems,
					 const float* __restrict__ data,
					 float* __restrict__ reluData,
					 float* __restrict__ derivative) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		reluData[i] = fmaxf(0.0f, data[i]);
		derivative[i] = !signbit(data[i]);
	}
}


extern "C" {
    cudaError_t _sigmoid(cudaStream_t stream,
                         int nelems,
			             const float* __restrict__ data,
			             float* __restrict__ sigmoid_data) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        sigmoid<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, data, sigmoid_data);
        return cudaGetLastError();
	}


	cudaError_t _sigmoidDer(cudaStream_t stream,
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


	cudaError_t _tanhDer(cudaStream_t stream,
                         int nelems,
			             const float* __restrict__ data,
			             float* __restrict__ tanh_data,
			             float* __restrict__ derivative) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        tanh<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, data, tanh_data, derivative);
        return cudaGetLastError();
	}

	cudaError_t _tanhSigm(cudaStream_t stream,
						  int axis,
                          int nrows,
                          int ncols,
			              const float* __restrict__ data,
			              float* __restrict__ tanh_sigm_data) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	    if (axis) {
	    	tanhSigmColumn<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, data, tanh_sigm_data);
	    } else {
	    	tanhSigmRow<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, data, tanh_sigm_data);
	    }
        return cudaGetLastError();
	}


	cudaError_t _tanhSigmDer(cudaStream_t stream,
							 int axis,
                             int nrows,
                             int ncols,
			                 const float* __restrict__ data,
			                 float* __restrict__ tanh_sigm_data,
			                 float* __restrict__ derivative) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
		if (axis) {
	    	tanhSigmColumn<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, data, tanh_sigm_data, derivative);
	    } else {
	    	tanhSigmRow<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, data, tanh_sigm_data, derivative);
	    }
        return cudaGetLastError();
	}


	cudaError_t _relu(cudaStream_t stream,
                      int nelems,
			          const float* __restrict__ data,
			          float* __restrict__ relu_data) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        relu<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, data, relu_data);
        return cudaGetLastError();
	}


	cudaError_t _reluDer(cudaStream_t stream,
                         int nelems,
			             const float* __restrict__ data,
			             float* __restrict__ relu_data,
			             float* __restrict__ derivative) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        relu<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, data, relu_data, derivative);
        return cudaGetLastError();
	}
}