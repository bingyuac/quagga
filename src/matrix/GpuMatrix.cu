#include "GpuMatrix.h"
#include "GpuMatrixKernels.cuh"
#include <algorithm>
#include <stdio.h>


GpuMatrix::GpuMatrix(int nrows, int ncols) : nrows(nrows), ncols(ncols), allocated(true) {
	init();
}


GpuMatrix::GpuMatrix(float *device_data, int nrows, int ncols) : data(device_data), nrows(nrows), ncols(ncols), allocated(false) {
	init();
}


GpuMatrix::GpuMatrix(const float *host_data, int nrows, int ncols) : nrows(nrows), ncols(ncols), allocated(true) {
	init();
	cublasSetVector(nelems, sizeof(float), host_data, 1, data, 1);
}


GpuMatrix::~GpuMatrix(void) {
	if (allocated) {
		cudaFree(data);
	}
}


GpuMatrix* GpuMatrix::dot(const GpuMatrix *other, GpuMatrixContext *context) {
	GpuMatrix *out = new GpuMatrix(nrows, other->ncols);
	GpuMatrix::dot(other, 0.0f, out, context);
	return out;
}


void GpuMatrix::dot(const GpuMatrix *other, GpuMatrix *out, GpuMatrixContext *context) {
	this->dot(other, 0.0f, out, context);
}


void GpuMatrix::dot(const GpuMatrix *other, float beta, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	if (other->ncols == 1) {
		cublasSgemv(context->get_cublas_handle(), CUBLAS_OP_N, nrows, ncols, &one, data, nrows, other->data, 1, &beta, out->data, 1);
	} else {
		cublasSgemm(context->get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, nrows, other->ncols, ncols, &one, data, nrows, other->data, other->nrows, &beta, out->data, out->nrows);
	}
}


void GpuMatrix::tdot(const GpuMatrix *other, GpuMatrix *out, GpuMatrixContext *context) {
	this->tdot(other, 0.0f, out, context);
}


void GpuMatrix::tdot(const GpuMatrix *other, float beta, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	if (other->ncols == 1) {
		cublasSgemv(context->get_cublas_handle(), CUBLAS_OP_T, nrows, ncols, &one, data, nrows, other->data, 1, &beta, out->data, 1);
	} else {
		cublasSgemm(context->get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, out->nrows, out->ncols, other->nrows, &one, data, nrows, other->data, other->nrows, &beta, out->data, out->nrows);
	}
}



float GpuMatrix::vdot(const GpuMatrix *other, GpuMatrixContext *context) {
	//	TODO Add special checking function
	float result;
	cublasSdot(context->get_cublas_handle(), nelems, data, 1, other->data, 1, &result);
	return result;
}


void GpuMatrix::tanh(GpuMatrix *tanh_matrix, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	kernels::tanh<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(nelems, data, tanh_matrix->data);
}


void GpuMatrix::tanh(GpuMatrix *tanh_matrix, GpuMatrix *derivative_matrix, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	kernels::tanh<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(nelems, data, tanh_matrix->data, derivative_matrix->data);
}


void GpuMatrix::sigmoid(GpuMatrix *sigmoid_matrix, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	kernels::sigmoid<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(nelems, data, sigmoid_matrix->data);
}


void GpuMatrix::sigmoid(GpuMatrix *sigmoid_matrix, GpuMatrix *derivative_matrix, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	kernels::sigmoid<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(nelems, data, sigmoid_matrix->data, derivative_matrix->data);
}


void GpuMatrix::hprod(const GpuMatrix *other, float alpha, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	kernels::hadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(nelems, data, other->data, alpha, out->data);
}


void GpuMatrix::hprod(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (out->nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	kernels::hadamardProduct<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(out->nelems, A->data, B->data, C->data, out->data);
}


void GpuMatrix::sumHprod(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, const GpuMatrix *D, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (out->nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	kernels::sumHprod<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(A->nelems, A->data, B->data, C->data, D->data, out->data);
}


void GpuMatrix::sumHprod(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, const GpuMatrix *D, const GpuMatrix *E, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (out->nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	kernels::sumHprod<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(A->nelems, A->data, B->data, C->data, D->data, E->data, out->data);
}


void GpuMatrix::sumHprod(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, const GpuMatrix *D, const GpuMatrix *E, const GpuMatrix *F, const GpuMatrix *G, const GpuMatrix *H, const GpuMatrix *I, const GpuMatrix *J, const GpuMatrix *K, GpuMatrix *out, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (out->nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	kernels::sumHprod<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(A->nelems, A->data, B->data, C->data, D->data, E->data, F->data, G->data, H->data, I->data, J->data, K->data, out->data);
}


void GpuMatrix::scale(float alpha, GpuMatrixContext *context) {
	cublasSscal(context->get_cublas_handle(), nelems, &alpha, data, 1);
}


void GpuMatrix::scale(float alpha, GpuMatrix *out, GpuMatrixContext *context) {
	int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	kernels::scale<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(nelems, data, alpha, out->data);
}


void GpuMatrix::add(const GpuMatrix *other, GpuMatrixContext *context) {
	GpuMatrix::add(1.0, other, context);
}


void GpuMatrix::add(float alpha, const GpuMatrix *other, GpuMatrixContext *context) {
	//	TODO Add special checking function
	cublasSaxpy(context->get_cublas_handle(), nelems, &alpha, other->data, 1, data, 1);
}


void GpuMatrix::add(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	kernels::sum<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(A->nelems, A->data, B->data, C->data, data, data);
}


void GpuMatrix::slicedAdd(const GpuMatrix *other, const int *column_indxs, GpuMatrixContext *context) {
	GpuMatrix::slicedAdd(other, column_indxs, 1.0, context);
}


void GpuMatrix::slicedAdd(const GpuMatrix *other, const int *column_indxs, float alpha, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int numBlocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	kernels::slicedInplaceAdd<<<numBlocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(other->nrows, other->ncols,  alpha, other->data, column_indxs, data);
}


GpuMatrix *GpuMatrix::sliceColumns(const int *column_indxs, int ncols, GpuMatrixContext *context) {
	GpuMatrix *out = new GpuMatrix(nrows, ncols);
	sliceColumns(out, column_indxs, context);
	return out;
}


void GpuMatrix::sliceColumns(const GpuMatrix *out, const int *column_indxs, GpuMatrixContext *context) {
	//	TODO Add special checking function
	int numBlocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL, (out->nelems - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
	kernels::sliceColumns<<<numBlocks, MAX_NUM_THREADS_PER_BLOCK, 0, context->get_cuda_stream()>>>(out->nrows, out->ncols, column_indxs, data, out->data);
}


float *GpuMatrix::flatten(float **matrix, int nrows, int ncols) {
	float *flatten_matrix = new float[nrows * ncols];
	for (int i = 0; i < ncols; i++) {
		for (int j = 0; j < nrows; j++) {
			flatten_matrix[j + i * nrows] = matrix[j][i];
		}
	}
	return flatten_matrix;
}


float **GpuMatrix::reshape(float *flatten_matrix, int nrows, int ncols) {
	float **matrix = new float*[nrows];
	for (int i = 0; i < nrows; i++) {
		matrix[i] = new float[ncols];
		for (int j = 0; j < ncols; j++) {
			matrix[i][j] = flatten_matrix[i + j * nrows];
		}
	}
	return matrix;
}


void GpuMatrix::print(float *flatten_matrix, int nrows, int ncols) {
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			printf("%.3f  ", flatten_matrix[i + j * nrows]);
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


float *GpuMatrix::toHost(GpuMatrixContext *context) {
	float *data = new float[nelems];
	cublasGetVectorAsync(nelems, sizeof(float), this->data, 1, data, 1, context->get_cuda_stream());
	return data;
}


void GpuMatrix::init() {
	nelems = nrows * ncols;
	nbytes = nelems * sizeof(float);
	if (allocated) {
		cudaMalloc(&data, nbytes);
	}
}


const float GpuMatrix::one = 1.0f;
