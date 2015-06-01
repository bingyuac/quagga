from quagga.matrix import GpuMatrix, CpuMatrix, CpuMatrixContext, \
    GpuMatrixContext
from quagga.matrix.CpuMatrix import CpuMatrix
from quagga.matrix.GpuMatrixContext import GpuMatrixContext
from quagga.matrix.CpuMatrixContext import CpuMatrixContext
MatrixClass = {'cpu': CpuMatrix, 'gpu': GpuMatrix}
MatrixContextClass = {'cpu': CpuMatrixContext, 'gpu': GpuMatrixContext}