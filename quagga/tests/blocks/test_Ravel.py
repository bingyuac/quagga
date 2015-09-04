import numpy as np
from unittest import TestCase
from quagga.blocks import Ravel
from quagga.context import Context
from quagga.matrix import GpuMatrix
from quagga.connector import Connector


class TestRavel(TestCase):
    def test_fprop(self):
        r = []
        n = 50

        for i in xrange(n):
            k = 4
            dim = 100
            matrix = Connector(GpuMatrix.from_npa(np.random.rand(dim, k), 'float'))
            ravel_block = Ravel(matrix)
            ravel_block.fprop()
            a = ravel_block.output.to_host()
            b = np.expand_dims(matrix.to_host().ravel(order='F'), axis=1)
            r.append(np.allclose(a, b))
        self.assertEqual(sum(r), n)

    def test_bprop(self):
        r = []
        n = 10

        context = Context()
        for i in xrange(n):
            k = 4
            dim = 100
            matrix = Connector(GpuMatrix.from_npa(np.random.rand(dim, k), 'float'), b_usage_context=context)
            ravel_block = Ravel(matrix)
            output, dL_doutput = ravel_block.output.register_usage(context, context)
            GpuMatrix.from_npa(np.random.rand(dL_doutput.nrows, dL_doutput.ncols), 'float').copy_to(context, dL_doutput)
            ravel_block.fprop()
            ravel_block.bprop()
            a = ravel_block.dL_dmatrix.to_host()
            b = dL_doutput.to_host().reshape(a.shape, order='F')
            r.append(np.allclose(a, b))
        self.assertEqual(sum(r), n)