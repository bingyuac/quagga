#include "GpuMatrix.h"
#include "GpuMatrixKernels.cuh"
#include <cublas_v2.h>
#include <stdio.h>


GpuMatrix::GpuMatrix(int nrows, int ncols) : nrows(nrows), ncols(ncols) {
	init(true);
}


GpuMatrix::GpuMatrix(float *device_data, int nrows, int ncols) : data(device_data), nrows(nrows), ncols(ncols) {
	init(false);
}


GpuMatrix::GpuMatrix(const float *host_data, int nrows, int ncols, GpuMatrixContext *context) : nrows(nrows), ncols(ncols) {
	init(true);
	cublasSetVectorAsync(nelems, sizeof(float), host_data, 1, data, 1, context->get_cuda_stream());
}


GpuMatrix::GpuMatrix(float *device_data, const float *host_data, int nrows, int ncols, GpuMatrixContext *context) : data(device_data), nrows(nrows), ncols(ncols) {
	init(false);
	cublasSetVectorAsync(nelems, sizeof(float), hostData, 1, data, 1, context->get_cuda_stream());
}


GpuMatrix::GpuMatrix(const GpuMatrix *other, GpuMatrixContext *context) : nrows(other->nrows), ncols(other->ncols) {
	init(true);
	cublasScopy(context->get_cublas_handle(), nelems, other->data, 1, data, 1);
}


GpuMatrix::~GpuMatrix(void) {
	if (allocated) {
		cudaFree(data);
	}
}


void GpuMatrix::dot(const GpuMatrix *other, GpuMatrix *out, GpuMatrixContext *context) {
	GpuMatrix::dot(other, 0.0f, out, context);
}


void GpuMatrix::dot(const GpuMatrix *other, float beta, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	if (other->ncols == 1) {
		cublasSgemv(context->get_cublas_handle(), CUBLAS_OP_N, nrows, ncols, &one, data, nrows, other->data, 1, &beta, out->data, 1);
	} else {
		cublasSgemm(context->get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, nrows, other->ncols, ncols, &one, data, nrows, other->data, other->nrows, &beta, out->data, out->nrows);
	}
}


void dot(const GpuMatrix *A, bool transpose, const GpuMatrix *B, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	cublasOperation_t transa = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
	if (other->ncols == 1) {
		cublasSgemv(context->get_cublas_handle(), transa, A->nrows, A->ncols, &one, A->data, A->nrows, B->data, 1, &beta, out->data, 1);
	} else {
		cublasSgemm(context->get_cublas_handle(), transa, CUBLAS_OP_N, out->nrows, out->ncols, B->nrows, &one, A->data, A->nrows, B->data, B->nrows, &beta, out->data, out->nrows);
	}
}


void hdot(const GpuMatrix *other, float alpha, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	int nelems = this->nrows * this->ncols;
	hadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(nelems, this->data, alpha, other->data, out->data);
}


void hdot(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	int nelems = out->nrows * out->ncols;
	hadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(nelems, A->data, B->data, C->data, out->data);
}


void GpuMatrix::tanh(GpuMatrix *tanh_matrix, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	tanh<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(nelems, data, tanh_matrix->data);
}


void GpuMatrix::tanh(GpuMatrix *tanh_matrix, GpuMatrix *derivative_matrix, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	tanh<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(nelems, data, tanh_matrix->data, derivative_matrix->data);
}


void sigmoid(GpuMatrix *sigmoid_matrix, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	sigmoid<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(nelems, data, sigmoid_matrix->data);
}


void sigmoid(GpuMatrix *sigmoid_matrix, GpuMatrix *derivative_matrix, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	sigmoid<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(nelems, data, sigmoid_matrix->data, derivative_matrix->data);
}


void addHdot(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, const GpuMatrix *D, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	addHdot<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(A->nelems, A->data, B->data, C->data, D->data, out->data);
}


void addHdot(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, const GpuMatrix *D, const GpuMatrix *E, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	addHdot<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(A->nelems, A->data, B->data, C->data, D->data, E->data, out->data);
}


void addHdot(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, const GpuMatrix *D, const GpuMatrix *E, const GpuMatrix *F, const GpuMatrix *G, const GpuMatrix *H, const GpuMatrix *I, const GpuMatrix *J, const GpuMatrix *K, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	addHdot<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(A->nelems, A->data, B->data, C->data, D->data, E->data, F->data, G->data, H->data, I->data, J->data, K->data, out->data);
}


float vdot(GpuMatrix *other, GpuMatrixContext *context) {
	//	TODO Add special checking function
	float result;
	cublasSdot(context->get_cublas_handle(), nelems, data, 1, other->data, 1, &result);
	return result;
}


void GpuMatrix::scale(float alpha, GpuMatrixContext *context) {
	cublasSscal(context->get_cublas_handle(), &alpha, nelems, data, 1);
}


void GpuMatrix::scale(float alpha, GpuMatrix *out, GpuMatrixContext *context) {
	int num_blocks = min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	scale<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(nelems, data, alpha, out->data);
}


void add(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	add<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(A->nelems, A->data, B->data, C->data, this->data, this->data);
}



//====================================



float *GpuMatrix::flatten(float **matrix, int nrows, int ncols) {
	float *flattenMatrix = new float[nrows * ncols];
	for (int i = 0; i < ncols; i++) {
		for (int j = 0; j < nrows; j++) {
			flattenMatrix[j + i * nrows] = matrix[j][i];
		}
	}
	return flattenMatrix;
}


float **GpuMatrix::reshape(float *flattenMatrix, int nrows, int ncols) {
	float **matrix = new float*[nrows];
	for (int i = 0; i < nrows; i++) {
		matrix[i] = new float[ncols];
		for (int j = 0; j < ncols; j++) {
			matrix[i][j] = flattenMatrix[i + j * nrows];
		}
	}
	return matrix;
}


void GpuMatrix::print(float *flattenMatrix, int nrows, int ncols) {
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			printf("%.3f  ", flattenMatrix[i + j * nrows]);
		}
		printf("\n");
	}
}


GpuMatrix **GpuMatrix::getColumns() {
	GpuMatrix ** columns = new GpuMatrix*[ncols];
	for (int i = 0; i < ncols; i++) {
		columns[i] = new GpuMatrix(&data[nrows * i], nrows, 1);
	}
	return columns;
}



void GpuMatrix::sliceColumns(const GpuMatrix *out, const int *columnIndxs, cudaStream_t stream) {
	//	TODO Add special checking function
	int nelems = out->nrows * out->ncols;
	int numBlocks = min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	sliceColumnsKernel<<<numBlocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(this->data, out->data, out->nrows, nelems, columnIndxs);
}


GpuMatrix *GpuMatrix::sliceColumns(const int *columnIndxs, int ncols, cudaStream_t stream) {
	GpuMatrix *out = new GpuMatrix(this->nrows, ncols);
	sliceColumns(out, columnIndxs, stream);
	return out;
}


void GpuMatrix::slicedAdd(const GpuMatrix *other, const int *columnIndxs, float alpha, cudaStream_t stream) {
	//	TODO Add special checking function
	int numBlocks = min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	slicedInplaceAddKernel<<<numBlocks, MAX_NUM_THREADS_PER_BLOCK, 0, stream>>>(other->nrows, other->ncols,  alpha, columnIndxs, other->data, this->data,);
}


void GpuMatrix::slicedAdd(const GpuMatrix *other, const int *columnIndxs, cudaStream_t stream) {
	GpuMatrix::slicedAdd(other, columnIndxs, 1.0, stream);
}


void GpuMatrix::add(const GpuMatrix *other, float alpha, cublasHandle_t cublasHandle) {
	//	TODO Add special checking function
	cublasSaxpy(cublasHandle, this->nelems, &alpha, other->data, 1, this->data, 1);
}


void GpuMatrix::add(const GpuMatrix *other, cublasHandle_t cublasHandle) {
	GpuMatrix::add(other, 1.0, cublasHandle);
}










float *GpuMatrix::toHost(cudaStream_t deviceToHostStream) {
	float *data = new float[this->nelems];
	cublasGetVectorAsync(this->nelems, sizeof(float), this->data, 1, data, 1, deviceToHostStream);
	return data;
}


void GpuMatrix::init(bool allocate_memory) {
	nelems = nrows * ncols;
	nbytes = nelems * sizeof(float);
	allocated = allocate_memory;
	if (allocate_memory) {
		cudaMalloc(&data, nbytes);
	}
}
