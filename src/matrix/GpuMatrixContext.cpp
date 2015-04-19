#include "GpuMatrixContext.h"


GpuMatrixContext::GpuMatrixContext(void) {
	cudaStreamCreate(&cuda_stream);
}


GpuMatrixContext::~GpuMatrixContext(void) {
	cudaStreamDestroy(cuda_stream);
}


void GpuMatrixContext::createCublasHandle(void) {
	cublasCreate(&GpuMatrixContext::cublas_handle);
}


void GpuMatrixContext::destroyCublasHandle(void) {
	cublasDestroy(cublas_handle);
}


cublasHandle_t GpuMatrixContext::get_cublas_handle() {
	cublasSetStream(cublas_handle, cuda_stream);
	return cublas_handle;
}


cudaStream_t GpuMatrixContext::get_cuda_stream() {
	return cuda_stream;
};


void GpuMatrixContext::synchronize(void) {
	cudaStreamSynchronize(cuda_stream);
}


cublasHandle_t GpuMatrixContext::cublas_handle;
