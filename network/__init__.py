from matrix import GpuMatrix, CpuMatrix, GpuMatrixContext, CpuMatrixContext
MatrixClass = {'cpu': CpuMatrix, 'gpu': GpuMatrix}
MatrixContextClass = {'cpu': CpuMatrixContext, 'gpu': GpuMatrixContext}
from LstmBlock import LstmBlock, MarginalLstmBlock