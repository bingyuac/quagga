#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include "../src/matrix/GpuMatrix.h"
#include "../src/matrix/GpuMatrixContext.h"


bool areSame(float a, float b) {
	return fabs(a - b) < 0.00001;
}


void testFlatten(void) {
	const int nrows = 2;
	const int ncols = 3;

	float matrix[nrows][ncols] = {
		{ 3.0,  4.3, 6.1},
		{-5.32, 9.1, 9.7}
	};
	float *matrixStartRows[nrows];
	for (int i = 0; i < nrows; i++) {
		matrixStartRows[i] = matrix[i];
	}
	float correctFlattenMatrix[nrows * ncols] = {3.0, -5.32, 4.3, 9.1, 6.1, 9.7};

	float *flattenArray = GpuMatrix::flatten(matrixStartRows, nrows, ncols);
	bool testPassed = true;
	for (int i = 0; i < nrows * ncols; i++) {
		if (!areSame(correctFlattenMatrix[i], flattenArray[i])) {
			testPassed = false;
			break;
		}
	}

	if (testPassed) {
		printf("testFlatten is passed\n");
	} else {
		printf("testFlatten is NOT passed!!!\n");
	}

	delete[] flattenArray;
}

void testReshape(void) {
	const int nrows = 2;
	const int ncols = 3;

	float flattenMatrix[nrows * ncols] = {3.0, -5.32, 4.3, 9.1, 6.1, 9.7};
	float correctMatrix[nrows][ncols] = {
		{ 3.0,  4.3, 6.1},
		{-5.32, 9.1, 9.7}
	};

	float **matrix = GpuMatrix::reshape(flattenMatrix, nrows, ncols);
	bool testPassed = true;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			if (!areSame(correctMatrix[i][j], matrix[i][j])) {
				testPassed = false;
				break;
			}
		}
	}

	if (testPassed) {
		printf("testReshape is passed\n");
	} else {
		printf("testReshape is NOT passed!!!\n");
	}

	for (int i = 0; i < nrows; i++) {
	    delete[] matrix[i];
	}
	delete[] matrix;
}


void testAdd(void) {
	const int nrows = 2;
	const int ncols = 3;

	float _A[nrows][ncols] = {
		{ 3.0,  4.3, 6.1},
		{-5.32, 9.1, 9.7}
	};
	float *matrixAStartRows[nrows];
	for (int i = 0; i < nrows; i++) {
		matrixAStartRows[i] = _A[i];
	}
	float _B[nrows][ncols] = {
		{-1.4, -1.3,  7.7},
		{ 8.2,  4.0, -0.9}
	};
	float *matrixBStartRows[nrows];
	for (int i = 0; i < nrows; i++) {
		matrixBStartRows[i] = _B[i];
	}
	float correctMatrix[nrows][ncols] = {
		{1.6,   3.0, 13.8},
		{2.88, 13.1,  8.8}
	};

	float *h_A = GpuMatrix::flatten(matrixAStartRows, nrows, ncols);
	float *h_B = GpuMatrix::flatten(matrixBStartRows, nrows, ncols);
	GpuMatrix *A = new GpuMatrix(h_A, nrows, ncols);
	GpuMatrix *B = new GpuMatrix(h_B, nrows, ncols);

	GpuMatrixContext *context = new GpuMatrixContext();
	A->add(B, context);
	float *flattenMatrix = A->toHost(context);
	context->synchronize();
	GpuMatrix::print(flattenMatrix, A->nrows, A->ncols);
	float **matrix = GpuMatrix::reshape(flattenMatrix, nrows, ncols);
	bool testPassed = true;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			if (!areSame(correctMatrix[i][j], matrix[i][j])) {
				testPassed = false;
				break;
			}
		}
	}

	if (testPassed) {
		printf("testAdd is passed\n");
	} else {
		printf("testAdd is NOT passed!!!\n");
	}

	delete[] h_A;
	delete[] h_B;
	delete context;
	cudaDeviceReset();
	delete A;
	delete B;
	delete[] flattenMatrix;
	for (int i = 0; i < nrows; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}


void testSlicedAdd(void){
	const int nrows = 2;
	const int aNcols = 4;
	const int bNcols = 3;

	float _A[nrows][aNcols] = {
		{ 3.0,  4.3, 6.1,  5.6},
		{-5.32, 9.1, 9.7, -4.42}
	};
	float *matrixAStartRows[nrows];
	for (int i = 0; i < nrows; i++) {
		matrixAStartRows[i] = _A[i];
	}
	float _B[nrows][bNcols] = {
		{-1.4, -1.3,  7.7},
		{ 8.2,  4.0, -0.9}
	};
	float *matrixBStartRows[nrows];
	for (int i = 0; i < nrows; i++) {
		matrixBStartRows[i] = _B[i];
	}
	int columnIndxsHost[bNcols] = {1, 1, 3};
	int *columnIndxs;
	cudaMalloc((void **)&columnIndxs, bNcols*sizeof(int));
	cublasSetVector(bNcols, sizeof(int), columnIndxsHost, 1, columnIndxs, 1);
	float correctMatrix[nrows][aNcols] = {
		{ 3.0,   1.6, 6.1, 13.3},
		{-5.32, 21.3, 9.7, -5.32}
	};

	float *h_A = GpuMatrix::flatten(matrixAStartRows, nrows, aNcols);
	float *h_B = GpuMatrix::flatten(matrixBStartRows, nrows, bNcols);
	GpuMatrix *A = new GpuMatrix(h_A, nrows, aNcols);
	GpuMatrix *B = new GpuMatrix(h_B, nrows, bNcols);
	GpuMatrixContext *context = new GpuMatrixContext();
	A->slicedAdd(B, columnIndxs, context);
	float *flattenMatrix = A->toHost(context);
	context->synchronize();
	float **matrix = GpuMatrix::reshape(flattenMatrix, nrows, aNcols);
	bool testPassed = true;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < aNcols; j++) {
			if (!areSame(correctMatrix[i][j], matrix[i][j])) {
				testPassed = false;
				break;
			}
		}
	}

	if (testPassed) {
		printf("testSlicedAdd is passed\n");
	} else {
		printf("testSlicedAdd is NOT passed!!!\n");
	}

	delete[] h_A;
	delete[] h_B;
	delete context;
	cudaDeviceReset();
	delete A;
	delete B;
	delete[] flattenMatrix;
	for (int i = 0; i < nrows; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}


void testDotColumnVector(void) {
	const int aNrows = 2;
	const int aNcols = 4;
	const int bNrows = 4;
	const int bNcols = 1;

	float _A[aNrows][aNcols] = {
		{ 3.0,  4.3, 6.1,  5.6},
		{-5.32, 9.1, 9.7, -4.42}
	};
	float *matrixAStartRows[aNrows];
	for (int i = 0; i < aNrows; i++) {
		matrixAStartRows[i] = _A[i];
	}
	float _B[bNrows][bNcols] = {
		{-1.4},
		{-1.3},
		{ 7.7},
		{ 8.2}
	};
	float *matrixBStartRows[bNrows];
	for (int i = 0; i < bNrows; i++) {
		matrixBStartRows[i] = _B[i];
	}
	float correctMatrix[bNrows][aNcols] = {
		{83.1},
		{34.064}
	};

	float *h_A = GpuMatrix::flatten(matrixAStartRows, aNrows, aNcols);
	float *h_B = GpuMatrix::flatten(matrixBStartRows, bNrows, bNcols);

	GpuMatrix *A = new GpuMatrix(h_A, aNrows, aNcols);
	GpuMatrix *B = new GpuMatrix(h_B, bNrows, bNcols);
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *out = A->dot(B, context);
	float *flattenMatrix = out->toHost(context);
	context->synchronize();
	float **matrix = GpuMatrix::reshape(flattenMatrix, out->nrows, out->ncols);
	bool testPassed = true;
	for (int i = 0; i < out->nrows; i++) {
		for (int j = 0; j < out->ncols; j++) {
			if (!areSame(correctMatrix[i][j], matrix[i][j])) {
				testPassed = false;
				break;
			}
		}
	}

	if (testPassed) {
		printf("testDotColumnVector is passed\n");
	} else {
		printf("testDotColumnVector is NOT passed!!!\n");
	}

	delete[] h_A;
	delete[] h_B;
	delete context;
	cudaDeviceReset();
	delete A;
	delete B;
	delete out;
	delete[] flattenMatrix;
	for (int i = 0; i < out->nrows; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}


void testDotMatrix(void) {
	const int aNrows = 2;
	const int aNcols = 4;
	const int bNrows = 4;
	const int bNcols = 3;

	float _A[aNrows][aNcols] = {
		{ 3.0,  4.3, 6.1,  5.6},
		{-5.32, 9.1, 9.7, -4.42}
	};
	float *matrixAStartRows[aNrows];
	for (int i = 0; i < aNrows; i++) {
		matrixAStartRows[i] = _A[i];
	}
	float _B[bNrows][bNcols] = {
		{-1.4,  5.6,  -3.1},
		{-1.3, -0.43,  5.3},
		{ 7.7,  7.12, -8.1},
		{ 8.2,  1.2,   5.94}
	};
	float *matrixBStartRows[bNrows];
	for (int i = 0; i < bNrows; i++) {
		matrixBStartRows[i] = _B[i];
	}
	float correctMatrix[aNrows][bNcols] = {
		{83.1,   65.103, -2.656},
		{34.064, 30.055, -40.1028}
	};

	float *h_A = GpuMatrix::flatten(matrixAStartRows, aNrows, aNcols);
	float *h_B = GpuMatrix::flatten(matrixBStartRows, bNrows, bNcols);


	GpuMatrix *A = new GpuMatrix(h_A, aNrows, aNcols);
	GpuMatrix *B = new GpuMatrix(h_B, bNrows, bNcols);
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *out = A->dot(B, context);
	float *flattenMatrix = out->toHost(context);
	context->synchronize();
	float **matrix = GpuMatrix::reshape(flattenMatrix, out->nrows, out->ncols);
	bool testPassed = true;
	for (int i = 0; i < out->nrows; i++) {
		for (int j = 0; j < out->ncols; j++) {
			if (!areSame(correctMatrix[i][j], matrix[i][j])) {
				testPassed = false;
				break;
			}
		}
	}

	if (testPassed) {
		printf("testDotMatrix is passed\n");
	} else {
		printf("testDotMatrix is NOT passed!!!\n");
	}

	delete[] h_A;
	delete[] h_B;
	delete context;
	cudaDeviceReset();
	delete A;
	delete B;
	delete out;
	delete[] flattenMatrix;
	for (int i = 0; i < out->nrows; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}


void testSliceColumns(void) {
	const int nrows = 2;
	const int aNcols = 4;
	const int bNcols = 3;

	float _A[nrows][aNcols] = {
		{ 3.0,  4.3, 6.1,  5.6},
		{-5.32, 9.1, 9.7, -4.42}
	};
	float *matrixAStartRows[nrows];
	for (int i = 0; i < nrows; i++) {
		matrixAStartRows[i] = _A[i];
	}

	int columnIndxsHost[bNcols] = {1, 1, 3};
	int *columnIndxs;
	cudaMalloc((void **)&columnIndxs, bNcols*sizeof(int));
	cublasSetVector(bNcols, sizeof(int), columnIndxsHost, 1, columnIndxs, 1);
	float correctMatrix[nrows][bNcols] = {
		{4.3, 4.3,  5.6},
		{9.1, 9.1, -4.42}
	};

	float *h_A = GpuMatrix::flatten(matrixAStartRows, nrows, aNcols);
	GpuMatrix *A = new GpuMatrix(h_A, nrows, aNcols);
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *B = A->sliceColumns(columnIndxs, bNcols, context);
	float *flattenMatrix = B->toHost(context);
	context->synchronize();
	float **matrix = GpuMatrix::reshape(flattenMatrix, nrows, bNcols);
	bool testPassed = true;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < bNcols; j++) {
			if (!areSame(correctMatrix[i][j], matrix[i][j])) {
				testPassed = false;
				break;
			}
		}
	}

	if (testPassed) {
		printf("testSliceColumns is passed\n");
	} else {
		printf("testSliceColumns is NOT passed!!!\n");
	}

	delete[] h_A;
	delete context;
	cudaDeviceReset();
	delete A;
	delete B;
	delete[] flattenMatrix;
	for (int i = 0; i < nrows; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}


int main(void) {
	GpuMatrixContext::createCublasHandle();
	testFlatten();
	testReshape();
	testAdd();
	testSlicedAdd();
	testDotColumnVector();
	testDotMatrix();
	testSliceColumns();
	GpuMatrixContext::destroyCublasHandle();
	return EXIT_SUCCESS;
}
