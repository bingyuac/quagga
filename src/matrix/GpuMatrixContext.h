#ifndef GPUMATRIXCONTEXT_H_
#define GPUMATRIXCONTEXT_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>


class GpuMatrixContext {
	public:
		GpuMatrixContext(void);
		~GpuMatrixContext(void);
		static void createCublasHandle(void);
		static void destroyCublasHandle(void);
		cublasHandle_t get_cublas_handle(void);
		cudaStream_t get_cuda_stream(void);
		void synchronize(void);
	private:
		static cublasHandle_t cublas_handle;
		cudaStream_t cuda_stream;
};


#endif
