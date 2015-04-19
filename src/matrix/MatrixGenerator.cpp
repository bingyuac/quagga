#include "MatrixGenerator.h"


MatrixGenerator::MatrixGenerator(int nrows, int ncols): nrows(nrows), ncols(ncols) {
}


GpuMatrix* MatrixGenerator::get_matrix(void) {
	return new GpuMatrix(nrows, ncols);
}
