#ifndef GPUMATRIX_H_
#define GPUMATRIX_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "matrix/GpuMatrixContext.h"


class GpuMatrix {
	public :
		GpuMatrix(int nrows, int ncols);
		GpuMatrix(float *device_data, int nrows, int ncols);
		GpuMatrix(const float *host_data, int nrows, int ncols, GpuMatrixContext *context);
		GpuMatrix(float *device_data, const float *host_data, int nrows, int ncols, GpuMatrixContext *context);
		GpuMatrix(const GpuMatrix *other, GpuMatrixContext *context);
		~GpuMatrix(void);
		void dot(const GpuMatrix *other, GpuMatrix *out, GpuMatrixContext *context);
		void dot(const GpuMatrix *other, float beta, GpuMatrix *out, GpuMatrixContext *context);
		static void dot(const GpuMatrix *A, bool transpose, const GpuMatrix *B, GpuMatrix *out, GpuMatrixContext *context);
		void hdot(const GpuMatrix *other, float beta, GpuMatrix *out, GpuMatrixContext *context); // C = A .* B + beta * C
		static void hdot(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, GpuMatrix *out, GpuMatrixContext *context); // out = A .* B .* C
		void tanh(GpuMatrix *tanh_matrix, GpuMatrixContext *context);
		void tanh(GpuMatrix *tanh_matrix, GpuMatrix *derivative_matrix, GpuMatrixContext *context);
		void sigmoid(GpuMatrix *sigmoid_matrix, GpuMatrixContext *context);
		void sigmoid(GpuMatrix *sigmoid_matrix, GpuMatrix *derivative_matrix, GpuMatrixContext *context);
		static void addHdot(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, const GpuMatrix *D, GpuMatrix *out, GpuMatrixContext *context); // out = A .* B + C .* D
		static void addHdot(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, const GpuMatrix *D, const GpuMatrix *E, GpuMatrix *out, GpuMatrixContext *context); // out = A .* B .* C + D .* E
		static void addHdot(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, const GpuMatrix *D, const GpuMatrix *E, const GpuMatrix *F, const GpuMatrix *G, const GpuMatrix *H, const GpuMatrix *I, const GpuMatrix *J, const GpuMatrix *K, GpuMatrix *out, GpuMatrixContext *context); // out = A .* B .* C + D .* E + F .* G + H .* I + J .* K
		float vdot(GpuMatrix *other, GpuMatrixContext *context);
		void scale(float alpha, GpuMatrixContext *context);
		void scale(float alpha, GpuMatrix *out, GpuMatrixContext *context);
		void add(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, const GpuMatrix *D, GpuMatrixContext *context);





		static float *flatten(float **matrix, int nrows, int ncols);
		static float **reshape(float *flattenMatrix, int nrows, int ncols);
		static void print(float *flattenMatrix, int nrows, int ncols);


		void sliceColumns(const GpuMatrix *out, const int *columnIndxs, cudaStream_t stream);
		GpuMatrix *sliceColumns(const int *columnIndxs, int ncols, cudaStream_t stream); // A[columnIndxs]
		GpuMatrix **getColumns();
		void slicedAdd(const GpuMatrix *other, const int *columnIndxs, float alpha, cudaStream_t stream); // A[columnIndxs] += alpha*B
		void slicedAdd(const GpuMatrix *other, const int *columnIndxs, cudaStream_t stream); // A[columnIndxs] += B
		void add(const GpuMatrix *other, float alpha, cublasHandle_t cublasHandle); // A += alpha*B
		void add(const GpuMatrix *other, cublasHandle_t cublasHandle); // A += B


		float *toHost(cudaStream_t deviceToHostStream);
	public:
		const int nrows;
		const int ncols;
		float *data;
	private:
		void init(bool allocateMemory);
		static const float one = 1.0f;
	private:
		int nelems;
		size_t nbytes;
		bool allocated;
};


#endif
