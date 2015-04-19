#ifndef MATRIXGENERATOR_H_
#define MATRIXGENERATOR_H_

#include "GpuMatrix.h"


class MatrixGenerator {
	public:
		MatrixGenerator(int nrows, int ncols);
		GpuMatrix* get_matrix(void);
		int nrows;
		int ncols;
};


#endif
