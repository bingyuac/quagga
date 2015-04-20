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


void testDotColumnVectorWithBeta(void) {
	const int a_nrows = 2;
	const int a_ncols = 4;
	const int b_nrows = 4;
	const int b_ncols = 1;
	float beta = 1.4f;

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
	float _out[a_nrows][b_ncols] = {
		{14.8},
		{-68.3}
	};
	float *matrix_out_start_rows[a_nrows];
	for (int i = 0; i < a_nrows; i++) {
		matrix_out_start_rows[i] = _out[i];
	}
	float correct_matrix[a_nrows][b_ncols] = {
		{103.82},
		{-61.556}
	};

	float *h_A = GpuMatrix::flatten(matrix_A_start_rows, a_nrows, a_ncols);
	float *h_B = GpuMatrix::flatten(matrix_B_start_rows, b_nrows, b_ncols);
	float *h_out= GpuMatrix::flatten(matrix_out_start_rows, a_nrows, b_ncols);
	GpuMatrixContext::createCublasHandle();
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *A = new GpuMatrix(h_A, a_nrows, a_ncols, context);
	GpuMatrix *B = new GpuMatrix(h_B, b_nrows, b_ncols, context);
	GpuMatrix *out = new GpuMatrix(h_out, a_nrows, b_ncols, context);

	A->dot(B, beta, out, context);
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
		printf("testDotColumnVectorWithBeta is passed\n");
	} else {
		printf("testDotColumnVectorWithBeta is NOT passed!!!\n");
	}

	delete[] h_A;
	delete[] h_B;
	delete[] h_out;
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


void testDotMatrixWithBeta(void) {
	const int a_nrows = 2;
	const int a_ncols = 4;
	const int b_nrows = 4;
	const int b_ncols = 3;
	float beta = -4.2f;

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
	float _out[a_nrows][b_ncols] = {
		{63.1,  5.13,  6.6},
		{-4.6, -8.55, -9.1}
	};
	float *matrix_out_start_rows[a_nrows];
	for (int i = 0; i < a_nrows; i++) {
		matrix_out_start_rows[i] = _out[i];
	}
	float correct_matrix[a_nrows][b_ncols] = {
		{-181.92,  43.557, -30.376},
		{  53.384, 65.965,  -1.8828}
	};

	float *h_A = GpuMatrix::flatten(matrix_A_start_rows, a_nrows, a_ncols);
	float *h_B = GpuMatrix::flatten(matrix_B_start_rows, b_nrows, b_ncols);
	float *h_out = GpuMatrix::flatten(matrix_out_start_rows, a_nrows, b_ncols);
	GpuMatrixContext::createCublasHandle();
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *A = new GpuMatrix(h_A, a_nrows, a_ncols, context);
	GpuMatrix *B = new GpuMatrix(h_B, b_nrows, b_ncols, context);
	GpuMatrix *out = new GpuMatrix(h_out, a_nrows, b_ncols, context);

	A->dot(B, beta, out, context);
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
		printf("testDotMatrixWithBeta is passed\n");
	} else {
		printf("testDotMatrixWithBeta is NOT passed!!!\n");
	}

	delete[] h_A;
	delete[] h_B;
	delete[] h_out;
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


void testTransposeDotColumnVectorWithBeta(void) {
	const int a_nrows = 2;
	const int a_ncols = 4;
	const int b_nrows = 2;
	const int b_ncols = 1;
	float beta = -0.13f;

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
		{ 1.3}
	};
	float *matrix_B_start_rows[b_nrows];
	for (int i = 0; i < b_nrows; i++) {
		matrix_B_start_rows[i] = _B[i];
	}
	float _out[a_ncols][b_ncols] = {
		{ 0.6},
		{ 9.8},
		{ 4.7},
		{-3.58}
	};
	float *matrix_out_start_rows[a_ncols];
	for (int i = 0; i < a_ncols; i++) {
		matrix_out_start_rows[i] = _out[i];
	}
	float correct_matrix[a_ncols][b_ncols] = {
		{-11.194},
		{  4.536},
		{  3.459},
		{-13.1206}
	};

	float *h_A = GpuMatrix::flatten(matrix_A_start_rows, a_nrows, a_ncols);
	float *h_B = GpuMatrix::flatten(matrix_B_start_rows, b_nrows, b_ncols);
	float *h_out= GpuMatrix::flatten(matrix_out_start_rows, a_ncols, b_ncols);
	GpuMatrixContext::createCublasHandle();
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *A = new GpuMatrix(h_A, a_nrows, a_ncols, context);
	GpuMatrix *B = new GpuMatrix(h_B, b_nrows, b_ncols, context);
	GpuMatrix *out = new GpuMatrix(h_out, a_ncols, b_ncols, context);

	A->tdot(B, beta, out, context);
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
		printf("testTransposeDotColumnVectorWithBeta is passed\n");
	} else {
		printf("testTransposeDotColumnVectorWithBeta is NOT passed!!!\n");
	}

	delete[] h_A;
	delete[] h_B;
	delete[] h_out;
	delete context;
	GpuMatrixContext::destroyCublasHandle();
	cudaDeviceReset();
	delete A;
	delete B;
	delete out;
	delete[] flatten_matrix;
	for (int i = 0; i < a_ncols; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}


void testTransposeDotMatrixWithBeta(void) {
	const int a_nrows = 2;
	const int a_ncols = 4;
	const int b_nrows = 2;
	const int b_ncols = 3;
	float beta = 0.2f;

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
		{-1.3, -0.43,  5.3}
	};
	float *matrix_B_start_rows[b_nrows];
	for (int i = 0; i < b_nrows; i++) {
		matrix_B_start_rows[i] = _B[i];
	}
	float _out[a_ncols][b_ncols] = {
		{  2.1,  9.8, -7.6},
		{-17.5,  0.6,  4.0},
		{-21.5, -9.8,  2.0},
		{ -2.9,  3.6, -0.6}
	};
	float *matrix_out_start_rows[a_ncols];
	for (int i = 0; i < a_ncols; i++) {
		matrix_out_start_rows[i] = _out[i];
	}
	float correct_matrix[a_ncols][b_ncols] = {
		{  3.136 ,  21.0476, -39.016},
		{-21.35  ,  20.287 ,  35.7  },
		{-25.45  ,  28.029 ,  32.9  },
		{ -2.674 ,  33.9806, -40.906}
	};

	float *h_A = GpuMatrix::flatten(matrix_A_start_rows, a_nrows, a_ncols);
	float *h_B = GpuMatrix::flatten(matrix_B_start_rows, b_nrows, b_ncols);
	float *h_out = GpuMatrix::flatten(matrix_out_start_rows, a_ncols, b_ncols);
	GpuMatrixContext::createCublasHandle();
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *A = new GpuMatrix(h_A, a_nrows, a_ncols, context);
	GpuMatrix *B = new GpuMatrix(h_B, b_nrows, b_ncols, context);
	GpuMatrix *out = new GpuMatrix(h_out, a_ncols, b_ncols, context);

	A->tdot(B, beta, out, context);
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
		printf("testTransposeDotMatrixWithBeta is passed\n");
	} else {
		printf("testTransposeDotMatrixWithBeta is NOT passed!!!\n");
	}

	delete[] h_A;
	delete[] h_B;
	delete[] h_out;
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


void testVectorDot(void) {
	const int a_nrows = 4;
	const int b_nrows = 4;

	float _A[a_nrows][1] = {
		{ 3.0},
		{ 4.3},
		{-6.1},
		{ 5.6}
	};
	float *matrix_A_start_rows[a_nrows];
	for (int i = 0; i < a_nrows; i++) {
		matrix_A_start_rows[i] = _A[i];
	}
	float _B[b_nrows][1] = {
		{-1.4},
		{-1.3},
		{ 7.7},
		{ 8.2}
	};
	float *matrix_B_start_rows[b_nrows];
	for (int i = 0; i < b_nrows; i++) {
		matrix_B_start_rows[i] = _B[i];
	}
	float target_value = -10.84;

	float *h_A = GpuMatrix::flatten(matrix_A_start_rows, a_nrows, 1);
	float *h_B = GpuMatrix::flatten(matrix_B_start_rows, b_nrows, 1);
	GpuMatrixContext::createCublasHandle();
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *A = new GpuMatrix(h_A, a_nrows, 1, context);
	GpuMatrix *B = new GpuMatrix(h_B, b_nrows, 1, context);
	float out = A->vdot(B, context);

	if (areSame(target_value, out)) {
		printf("testVectorDot is passed\n");
	} else {
		printf("testVectorDot is NOT passed!!!\n");
	}

	delete[] h_A;
	delete[] h_B;
	delete context;
	GpuMatrixContext::destroyCublasHandle();
	cudaDeviceReset();
	delete A;
	delete B;
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


void testTanh(void) {
	const int a_nrows = 2;
	const int a_ncols = 4;

	float _A[a_nrows][a_ncols] = {
		{ 0.5,  0.3,  1.1,  2.6},
		{-5.32, 0.1, -9.7, -0.42}
	};
	float *matrix_A_start_rows[a_nrows];
	for (int i = 0; i < a_nrows; i++) {
		matrix_A_start_rows[i] = _A[i];
	}
	float _tanh[a_nrows][a_ncols];
	float *matrix_tanh_start_rows[a_ncols];
	for (int i = 0; i < a_ncols; i++) {
		matrix_tanh_start_rows[i] = _tanh[i];
	}
	float correct_matrix[a_nrows][a_ncols] = {
		{ 0.46211716,  0.29131261,  0.80049902,  0.9890274 },
		{-0.99995212,  0.09966799, -0.99999999, -0.39693043},
	};

	float *h_A = GpuMatrix::flatten(matrix_A_start_rows, a_nrows, a_ncols);
	float *h_tanh = GpuMatrix::flatten(matrix_tanh_start_rows, a_nrows, a_ncols);
	GpuMatrixContext::createCublasHandle();
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *A = new GpuMatrix(h_A, a_nrows, a_ncols, context);
	GpuMatrix *tanh = new GpuMatrix(h_tanh, a_nrows, a_ncols, context);

	A->tanh(tanh, context);
	float *flatten_matrix = tanh->toHost(context);
	context->synchronize();
	float **matrix = GpuMatrix::reshape(flatten_matrix, tanh->nrows, tanh->ncols);
	bool test_passed = true;
	for (int i = 0; i < tanh->nrows; i++) {
		for (int j = 0; j < tanh->ncols; j++) {
			if (!areSame(correct_matrix[i][j], matrix[i][j])) {
				test_passed = false;
				break;
			}
		}
	}

	if (test_passed) {
		printf("testTanh is passed\n");
	} else {
		printf("testTanh is NOT passed!!!\n");
	}

	delete[] h_A;
	delete[] h_tanh;
	delete context;
	GpuMatrixContext::destroyCublasHandle();
	cudaDeviceReset();
	delete A;
	delete tanh;
	delete[] flatten_matrix;
	for (int i = 0; i < a_nrows; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}


void testTanhWithDerivatives(void) {
	const int a_nrows = 2;
	const int a_ncols = 4;

	float _A[a_nrows][a_ncols] = {
		{ 0.5,  0.3,  1.1,  2.6},
		{-5.32, 0.1, -9.7, -0.42}
	};
	float *matrix_A_start_rows[a_nrows];
	for (int i = 0; i < a_nrows; i++) {
		matrix_A_start_rows[i] = _A[i];
	}
	float _tanh[a_nrows][a_ncols];
	float *matrix_tanh_start_rows[a_ncols];
	for (int i = 0; i < a_ncols; i++) {
		matrix_tanh_start_rows[i] = _tanh[i];
	}
	float _tanh_deriv[a_nrows][a_ncols];
	float *matrix_tanh_deriv_start_rows[a_ncols];
	for (int i = 0; i < a_ncols; i++) {
		matrix_tanh_deriv_start_rows[i] = _tanh[i];
	}
	float correct_tanh_matrix[a_nrows][a_ncols] = {
		{ 0.46211716,  0.29131261,  0.80049902,  0.9890274 },
		{-0.99995212,  0.09966799, -0.99999999, -0.39693043},
	};
	float correct_tanh_deriv_matrix[a_nrows][a_ncols] = {
		{7.86447733e-01,   9.15136962e-01,   3.59201316e-01, 2.18247977e-02},
		{9.57515716e-05,   9.90066291e-01,   1.50226670e-08, 8.42446232e-01}
	};


	float *h_A = GpuMatrix::flatten(matrix_A_start_rows, a_nrows, a_ncols);
	float *h_tanh = GpuMatrix::flatten(matrix_tanh_start_rows, a_nrows, a_ncols);
	float *h_tanh_deriv = GpuMatrix::flatten(matrix_tanh_deriv_start_rows, a_nrows, a_ncols);
	GpuMatrixContext::createCublasHandle();
	GpuMatrixContext *context = new GpuMatrixContext();
	GpuMatrix *A = new GpuMatrix(h_A, a_nrows, a_ncols, context);
	GpuMatrix *tanh = new GpuMatrix(h_tanh, a_nrows, a_ncols, context);
	GpuMatrix *tanh_deriv = new GpuMatrix(h_tanh_deriv, a_nrows, a_ncols, context);

	A->tanh(tanh, tanh_deriv, context);
	float *flatten_tanh_matrix = tanh->toHost(context);
	float *flatten_tanh_deriv_matrix = tanh_deriv->toHost(context);
	context->synchronize();
	float **tanh_matrix = GpuMatrix::reshape(flatten_tanh_matrix, tanh->nrows, tanh->ncols);
	float **tanh_deriv_matrix = GpuMatrix::reshape(flatten_tanh_deriv_matrix, tanh_deriv->nrows, tanh_deriv->ncols);
	bool test_passed = true;
	for (int i = 0; i < tanh->nrows; i++) {
		for (int j = 0; j < tanh->ncols; j++) {
			if (!areSame(correct_tanh_matrix[i][j], tanh_matrix[i][j]) ||
				!areSame(correct_tanh_deriv_matrix[i][j], tanh_deriv_matrix[i][j])) {
				test_passed = false;
				break;
			}
		}
	}

	if (test_passed) {
		printf("testTanhWithDerivatives is passed\n");
	} else {
		printf("testTanhWithDerivatives is NOT passed!!!\n");
	}

	delete[] h_A;
	delete[] h_tanh;
	delete[] h_tanh_deriv;
	delete context;
	GpuMatrixContext::destroyCublasHandle();
	cudaDeviceReset();
	delete A;
	delete tanh;
	delete tanh_deriv;
	delete[] flatten_tanh_matrix;
	for (int i = 0; i < a_nrows; i++) {
		delete[] tanh_matrix[i];
	}
	delete[] tanh_matrix;
	delete[] flatten_tanh_deriv_matrix;
	for (int i = 0; i < a_nrows; i++) {
		delete[] tanh_deriv_matrix[i];
	}
	delete[] tanh_deriv_matrix;
}


void testSigmoid(void) {

}


void testTanhWothDerivatives(void) {

}


int main(void) {
	testFlatten();
	testReshape();
	testAdd();
	testSlicedAdd();
	testDotColumnVector();
	testDotColumnVectorWithBeta();
	testDotMatrix();
	testDotMatrixWithBeta();
	testTransposeDotColumnVectorWithBeta();
	testTransposeDotMatrixWithBeta();
	testVectorDot();
	testTanh();
	testTanhWithDerivatives();
	testSliceColumns();
	return EXIT_SUCCESS;
}
