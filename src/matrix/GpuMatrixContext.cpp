#include "GpuMatrixContext.h"
#include <stdio.h>
//TODO


GpuMatrixContext::GpuMatrixContext(void) {
	int n = cudaStreamCreate(&cuda_stream);
	printf("cudaStreamCreate %d\n", n);
}


GpuMatrixContext::~GpuMatrixContext(void) {
	int n = cudaStreamDestroy(cuda_stream);
	printf("cudaStreamDestroy %d\n", n);
}


void GpuMatrixContext::createCublasHandle(void) {
	int n = cublasCreate(&GpuMatrixContext::cublas_handle);
	printf("createCublasHandle %d\n", n);
}


void GpuMatrixContext::destroyCublasHandle(void) {
	cublasDestroy(cublas_handle);
}


cublasHandle_t GpuMatrixContext::get_cublas_handle() {
	int n = cublasSetStream(cublas_handle, cuda_stream);
	printf("get_cublas_handle %d\n", n);
	return cublas_handle;
}


cudaStream_t GpuMatrixContext::get_cuda_stream() {
	return cuda_stream;
};


void GpuMatrixContext::synchronize(void) {
	int n = cudaStreamSynchronize(cuda_stream);
	printf("cudaStreamSynchronize %d\n", n);
}


cublasHandle_t GpuMatrixContext::cublas_handle;
