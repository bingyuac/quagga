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
	float *matrix_start_rows[nrows];
	for (int i = 0; i < nrows; i++) {
		matrix_start_rows[i] = matrix[i];
	}
	float correct_flatten_matrix[nrows * ncols] = {3.0, -5.32, 4.3, 9.1, 6.1, 9.7};

	float *flatten_array = GpuMatrix::flatten(matrix_start_rows, nrows, ncols);
	bool test_passed = true;
	for (int i = 0; i < nrows * ncols; i++) {
		if (!areSame(correct_flatten_matrix[i], flatten_array[i])) {
			test_passed = false;
			break;
		}
	}

	if (test_passed) {
		printf("testFlatten is passed\n");
	} else {
		printf("testFlatten is NOT passed!!!\n");
	}

	delete[] flatten_array;
}

void testReshape(void) {
	const int nrows = 2;
	const int ncols = 3;

	float flatten_matrix[nrows * ncols] = {3.0, -5.32, 4.3, 9.1, 6.1, 9.7};
	float correct_matrix[nrows][ncols] = {
		{ 3.0,  4.3, 6.1},
		{-5.32, 9.1, 9.7}
	};

	float **matrix = GpuMatrix::reshape(flatten_matrix, nrows, ncols);
	bool test_passed = true;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			if (!areSame(correct_matrix[i][j], matrix[i][j])) {
				test_passed = false;
				break;
			}
		}
	}

	if (test_passed) {
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
	float *matrix_A_start_rows[nrows];
	for (int i = 0; i < nrows; i++) {
		matrix_A_start_rows[i] = _A[i];
	}
	float _B[nrows][ncols] = {
		{-1.4, -1.3,  7.7},
		{ 8.2,  4.0, -0.9}
	};
	float *matrix_B_start_rows[nrows];
	for (int i = 0; i < nrows; i++) {
		matrix_B_start_rows[i] = _B[i];
	}
	float correct_matrix[nrows][ncols] = {
		{1.6,   3.0, 13.8},
		{2.88, 13.1,  8.8}
	};

	float *h_A = GpuMatrix::flatten(matrix_A_start_rows, nrows, ncols);
	float *h_B = GpuMatrix::flatten(matrix_B_start_rows, nrows, ncols);
	GpuMatrixContext::createCublasHandle();
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *A = new GpuMatrix(h_A, nrows, ncols, context);
	GpuMatrix *B = new GpuMatrix(h_B, nrows, ncols, context);

	A->add(B, context);
	float *flatten_matrix = A->toHost(context);
	context->synchronize();
	float **matrix = GpuMatrix::reshape(flatten_matrix, nrows, ncols);
	bool testPassed = true;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			if (!areSame(correct_matrix[i][j], matrix[i][j])) {
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
	GpuMatrixContext::destroyCublasHandle();
	cudaDeviceReset();
	delete A;
	delete B;
	delete[] flatten_matrix;
	for (int i = 0; i < nrows; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}


void testSlicedAdd(void){
	const int nrows = 2;
	const int a_ncols = 4;
	const int b_ncols = 3;

	float _A[nrows][a_ncols] = {
		{ 3.0,  4.3, 6.1,  5.6},
		{-5.32, 9.1, 9.7, -4.42}
	};
	float *matrix_A_start_rows[nrows];
	for (int i = 0; i < nrows; i++) {
		matrix_A_start_rows[i] = _A[i];
	}
	float _B[nrows][b_ncols] = {
		{-1.4, -1.3,  7.7},
		{ 8.2,  4.0, -0.9}
	};
	float *matrix_B_start_rows[nrows];
	for (int i = 0; i < nrows; i++) {
		matrix_B_start_rows[i] = _B[i];
	}
	int column_indxs_host[b_ncols] = {1, 1, 3};
	int *column_indxs;
	cudaMalloc((void **)&column_indxs, b_ncols*sizeof(int));
	cublasSetVector(b_ncols, sizeof(int), column_indxs_host, 1, column_indxs, 1);
	float correct_matrix[nrows][a_ncols] = {
		{ 3.0,   1.6, 6.1, 13.3},
		{-5.32, 21.3, 9.7, -5.32}
	};

	float *h_A = GpuMatrix::flatten(matrix_A_start_rows, nrows, a_ncols);
	float *h_B = GpuMatrix::flatten(matrix_B_start_rows, nrows, b_ncols);
	GpuMatrixContext::createCublasHandle();
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *A = new GpuMatrix(h_A, nrows, a_ncols, context);
	GpuMatrix *B = new GpuMatrix(h_B, nrows, b_ncols, context);

	A->slicedAdd(B, column_indxs, context);
	float *flatten_matrix = A->toHost(context);
	context->synchronize();
	float **matrix = GpuMatrix::reshape(flatten_matrix, nrows, a_ncols);
	bool testPassed = true;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < a_ncols; j++) {
			if (!areSame(correct_matrix[i][j], matrix[i][j])) {
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
	GpuMatrixContext::destroyCublasHandle();
	cudaDeviceReset();
	delete A;
	delete B;
	delete[] flatten_matrix;
	for (int i = 0; i < nrows; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}


void testDotColumnVector(void) {
	const int a_nrows = 2;
	const int a_ncols = 4;
	const int b_nrows = 4;
	const int b_ncols = 1;

	float _A[a_nrows][a_ncols] = {
		{ 3.0,  4.3, 6.1,  5.6},
		{-5.32, 9.1, 9.7, -4.42}
	};
	float *matrix_A_start_rows[a_nrows];
	for (int i = 0; i < a_nrows; i++) {
		matrix_A_start_rows[i] = _A[i];
	}
	float _B[b_nrows][b_ncols] = {
		{-1.4},
		{-1.3},
		{ 7.7},
		{ 8.2}
	};
	float *matrix_B_start_rows[b_nrows];
	for (int i = 0; i < b_nrows; i++) {
		matrix_B_start_rows[i] = _B[i];
	}
	float correct_matrix[a_nrows][b_ncols] = {
		{83.1},
		{34.064}
	};

	float *h_A = GpuMatrix::flatten(matrix_A_start_rows, a_nrows, a_ncols);
	float *h_B = GpuMatrix::flatten(matrix_B_start_rows, b_nrows, b_ncols);
	GpuMatrixContext::createCublasHandle();
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *A = new GpuMatrix(h_A, a_nrows, a_ncols, context);
	GpuMatrix *B = new GpuMatrix(h_B, b_nrows, b_ncols, context);

	GpuMatrix *out = A->dot(B, context);
	float *flatten_matrix = out->toHost(context);
	context->synchronize();
	float **matrix = GpuMatrix::reshape(flatten_matrix, out->nrows, out->ncols);
	bool test_passed = true;
	for (int i = 0; i < out->nrows; i++) {
		for (int j = 0; j < out->ncols; j++) {
			if (!areSame(correct_matrix[i][j], matrix[i][j])) {
				test_passed = false;
				break;
			}
		}
	}

	if (test_passed) {
		printf("testDotColumnVector is passed\n");
	} else {
		printf("testDotColumnVector is NOT passed!!!\n");
	}

	delete[] h_A;
	delete[] h_B;
	delete context;
	GpuMatrixContext::destroyCublasHandle();
	cudaDeviceReset();
	delete A;
	delete B;
	delete out;
	delete[] flatten_matrix;
	for (int i = 0; i < a_nrows; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}


void testDotMatrix(void) {
	const int a_nrows = 2;
	const int a_ncols = 4;
	const int b_nrows = 4;
	const int b_ncols = 3;

	float _A[a_nrows][a_ncols] = {
		{ 3.0,  4.3, 6.1,  5.6},
		{-5.32, 9.1, 9.7, -4.42}
	};
	float *matrix_A_start_rows[a_nrows];
	for (int i = 0; i < a_nrows; i++) {
		matrix_A_start_rows[i] = _A[i];
	}
	float _B[b_nrows][b_ncols] = {
		{-1.4,  5.6,  -3.1},
		{-1.3, -0.43,  5.3},
		{ 7.7,  7.12, -8.1},
		{ 8.2,  1.2,   5.94}
	};
	float *matrix_B_start_rows[b_nrows];
	for (int i = 0; i < b_nrows; i++) {
		matrix_B_start_rows[i] = _B[i];
	}
	float correct_matrix[a_nrows][b_ncols] = {
		{83.1,   65.103, -2.656},
		{34.064, 30.055, -40.1028}
	};

	float *h_A = GpuMatrix::flatten(matrix_A_start_rows, a_nrows, a_ncols);
	float *h_B = GpuMatrix::flatten(matrix_B_start_rows, b_nrows, b_ncols);
	GpuMatrixContext::createCublasHandle();
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *A = new GpuMatrix(h_A, a_nrows, a_ncols, context);
	GpuMatrix *B = new GpuMatrix(h_B, b_nrows, b_ncols, context);

	GpuMatrix *out = A->dot(B, context);
	float *flatten_matrix = out->toHost(context);
	context->synchronize();
	float **matrix = GpuMatrix::reshape(flatten_matrix, out->nrows, out->ncols);
	bool test_passed = true;
	for (int i = 0; i < out->nrows; i++) {
		for (int j = 0; j < out->ncols; j++) {
			if (!areSame(correct_matrix[i][j], matrix[i][j])) {
				test_passed = false;
				break;
			}
		}
	}

	if (test_passed) {
		printf("testDotMatrix is passed\n");
	} else {
		printf("testDotMatrix is NOT passed!!!\n");
	}

	delete[] h_A;
	delete[] h_B;
	delete context;
	GpuMatrixContext::destroyCublasHandle();
	cudaDeviceReset();
	delete A;
	delete B;
	delete out;
	delete[] flatten_matrix;
	for (int i = 0; i < a_nrows; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}


void testSliceColumns(void) {
	const int nrows = 2;
	const int a_ncols = 4;
	const int b_ncols = 3;

	float _A[nrows][a_ncols] = {
		{ 3.0,  4.3, 6.1,  5.6},
		{-5.32, 9.1, 9.7, -4.42}
	};
	float *matrix_A_start_rows[nrows];
	for (int i = 0; i < nrows; i++) {
		matrix_A_start_rows[i] = _A[i];
	}

	int column_indxs_host[b_ncols] = {1, 1, 3};
	int *column_indxs;
	cudaMalloc((void **)&column_indxs, b_ncols*sizeof(int));
	cublasSetVector(b_ncols, sizeof(int), column_indxs_host, 1, column_indxs, 1);
	float correct_matrix[nrows][b_ncols] = {
		{4.3, 4.3,  5.6},
		{9.1, 9.1, -4.42}
	};

	float *h_A = GpuMatrix::flatten(matrix_A_start_rows, nrows, a_ncols);
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *A = new GpuMatrix(h_A, nrows, a_ncols, context);
	GpuMatrix *B = A->sliceColumns(column_indxs, b_ncols, context);
	float *flattenMatrix = B->toHost(context);
	context->synchronize();
	float **matrix = GpuMatrix::reshape(flattenMatrix, nrows, b_ncols);
	bool test_passed = true;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < b_ncols; j++) {
			if (!areSame(correct_matrix[i][j], matrix[i][j])) {
				test_passed = false;
				break;
			}
		}
	}

	if (test_passed) {
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
	testFlatten();
	testReshape();
	testAdd();
	testSlicedAdd();
	testDotColumnVector();
	testDotMatrix();
	testSliceColumns();
	return EXIT_SUCCESS;
}
