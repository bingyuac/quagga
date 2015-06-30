#include <algorithm>
#include <cuda_runtime.h>


#define MAX_NUM_THREADS_PER_BLOCK 512
#define MAX_NUM_BLOCKS_PER_KERNEL 64


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


__global__ void tanhSigm(int nrows,
						 int ncols,
						 const float* __restrict__ data,
						 float* __restrict__ sigmTanhData) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nelems = nrows * ncols;
	const int margin  = 3 * nrows / 4;

	for (int i = start_i; i < nelems; i += nthreads) {
		if (i % nrows < margin) {
			sigmTanhData[i] = 1.0f / (1.0f + expf(-data[i]));
		} else {
			sigmTanhData[i] = tanhf(data[i]);
		}
	}
}


__global__ void tanhSigm(int nrows,
						 int ncols,
						 const float* __restrict__ data,
						 float* __restrict__ sigmTanhData,
						 float* __restrict__ derivative) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int nelems = nrows * ncols;
	const int margin  = 3 * nrows / 4;

	for (int i = start_i; i < nelems; i += nthreads) {
		if (i % nrows < margin) {
			sigmTanhData[i] = 1.0f / (1.0f + expf(-data[i]));
			derivative[i] = sigmTanhData[i] * (1.0f - sigmTanhData[i]);
		} else {
			sigmTanhData[i] = tanhf(data[i]);
			derivative[i] = 1.0f - sigmTanhData[i] * sigmTanhData[i];
		}
	}
}


__global__ void relu(int nelems,
				 	 const float* __restrict__ data,
					 float* __restrict__ reluData) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		reluData[i] = fmaxf(0.0, data[i]);
	}
}


__global__ void relu(int nelems,
					 const float* __restrict__ data,
					 float* __restrict__ reluData,
					 float* __restrict__ derivative) {
	const int nthreads = blockDim.x * gridDim.x;
	const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = start_i; i < nelems; i += nthreads) {
		reluData[i] = fmaxf(0.0, data[i]);
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

	cudaError_t _tanh_sigm(cudaStream_t stream,
                           int nrows,
                           int ncols,
			               const float* __restrict__ data,
			               float* __restrict__ sigm_tanh_data) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        tanhSigm<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, data, sigm_tanh_data);
        return cudaGetLastError();
	}


	cudaError_t _tanh_sigm_der(cudaStream_t stream,
                               int nrows,
                               int ncols,
			                   const float* __restrict__ data,
			                   float* __restrict__ sigm_tanh_data,
			                   float* __restrict__ derivative) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nrows * ncols - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        tanhSigm<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nrows, ncols, data, sigm_tanh_data, derivative);
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


	cudaError_t _relu_der(cudaStream_t stream,
                          int nelems,
			              const float* __restrict__ data,
			              float* __restrict__ relu_data,
			              float* __restrict__ derivative) {
	    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
        relu<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(nelems, data, relu_data, derivative);
        return cudaGetLastError();
	}
}