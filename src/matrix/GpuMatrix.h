#ifndef GPUMATRIX_H_
#define GPUMATRIX_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "GpuMatrixContext.h"


class GpuMatrix {
	public :
		GpuMatrix(int nrows, int ncols);
		GpuMatrix(float *device_data, int nrows, int ncols);
		GpuMatrix(const float *host_data, int nrows, int ncols);
		~GpuMatrix(void);

		GpuMatrix* dot(const GpuMatrix *other, GpuMatrixContext *context);
		void dot(const GpuMatrix *other, GpuMatrix *out, GpuMatrixContext *context);
		void dot(const GpuMatrix *other, float beta, GpuMatrix *out, GpuMatrixContext *context); // out = this * other + beta * out
		void tdot(const GpuMatrix *other, GpuMatrix *out, GpuMatrixContext *context);
		void tdot(const GpuMatrix *other, float beta, GpuMatrix *out, GpuMatrixContext *context); // out = this.T * other + beta * out
		float vdot(const GpuMatrix *other, GpuMatrixContext *context);

		void tanh(GpuMatrix *tanh_matrix, GpuMatrixContext *context);
		void tanh(GpuMatrix *tanh_matrix, GpuMatrix *derivative_matrix, GpuMatrixContext *context);
		void sigmoid(GpuMatrix *sigmoid_matrix, GpuMatrixContext *context);
		void sigmoid(GpuMatrix *sigmoid_matrix, GpuMatrix *derivative_matrix, GpuMatrixContext *context);

		void hprod(const GpuMatrix *other, float beta, GpuMatrix *out, GpuMatrixContext *context); // out = this .* other + beta * out
		static void hprod(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, GpuMatrix *out, GpuMatrixContext *context); // out = A .* B .* C
		static void sumHprod(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, const GpuMatrix *D, GpuMatrix *out, GpuMatrixContext *context); // out = A .* B + C .* D
		static void sumHprod(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, const GpuMatrix *D, const GpuMatrix *E, GpuMatrix *out, GpuMatrixContext *context); // out = A .* B .* C + D .* E
		static void sumHprod(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, const GpuMatrix *D, const GpuMatrix *E, const GpuMatrix *F, const GpuMatrix *G, const GpuMatrix *H, const GpuMatrix *I, const GpuMatrix *J, const GpuMatrix *K, GpuMatrix *out, GpuMatrixContext *context); // out = A .* B .* C + D .* E + F .* G + H .* I + J .* K

		void scale(float alpha, GpuMatrixContext *context);
		void scale(float alpha, GpuMatrix *out, GpuMatrixContext *context);

		void add(const GpuMatrix *other, GpuMatrixContext *context); // A += B
		void add(float alpha, const GpuMatrix *other, GpuMatrixContext *context); // A += alpha*B
		void add(const GpuMatrix *A, const GpuMatrix *B, const GpuMatrix *C, GpuMatrixContext *context); // this += A + B + C
		void slicedAdd(const GpuMatrix *other, const int *column_indxs, GpuMatrixContext *context); // A[columnIndxs] += B
		void slicedAdd(const GpuMatrix *other, const int *column_indxs, float alpha, GpuMatrixContext *context); // A[columnIndxs] += alpha*B

		GpuMatrix *sliceColumns(const int *column_indxs, int ncols, GpuMatrixContext *context); // A[columnIndxs]
		void sliceColumns(const GpuMatrix *out, const int *column_indxs, GpuMatrixContext *context);
		GpuMatrix **getColumns();

		static float *flatten(float **matrix, int nrows, int ncols);
		static float **reshape(float *flatten_matrix, int nrows, int ncols);
		static void print(float *flatten_matrix, int nrows, int ncols);
		float *toHost(GpuMatrixContext *context);
	public:
		const int nrows;
		const int ncols;
		float *data;
	private:
		void init();
	private:
		int nelems;
		size_t nbytes;
		bool allocated;
		static const float one;
};


#endif
