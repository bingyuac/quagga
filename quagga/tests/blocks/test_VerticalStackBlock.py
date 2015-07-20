import numpy as np
from itertools import izip
from unittest import TestCase
from quagga.context import Context
from quagga.matrix import GpuMatrix
from quagga.connector import Connector
from quagga.blocks import VerticalStackBlock


class TestVerticalStackBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 50

    def test_fprop(self):
        r = []

        for i in xrange(self.N):
            matrices = []
            ncols = self.rng.random_integers(1, 5000)
            for _ in xrange(self.rng.random_integers(1, 10)):
                nrows = self.rng.random_integers(1, 5000)
                matrices.append(Connector(GpuMatrix.from_npa(np.random.rand(nrows, ncols), 'float')))
            vstack_block = VerticalStackBlock(*matrices)
            vstack_block.fprop()
            a = vstack_block.output.to_host()
            b = np.vstack([matrix.to_host() for matrix in matrices])
            r.append(np.allclose(a, b))

        self.assertEqual(sum(r), self.N)

    def test_bprop(self):
        r = []

        context = Context()
        for i in xrange(self.N):
            matrices = []
            ncols = self.rng.random_integers(1, 3000)
            nrows = [0]
            row_slices = []
            for _ in xrange(self.rng.random_integers(1, 10)):
                _nrows = self.rng.random_integers(1, 2000)
                nrows.append(nrows[-1] + _nrows)
                if self.rng.choice([True, False]):
                    b_usage_context = context
                    row_slices.append((nrows[-2], nrows[-1]))
                else:
                    b_usage_context = None
                matrices.append(Connector(GpuMatrix.from_npa(np.random.rand(_nrows, ncols), 'float'), b_usage_context=b_usage_context))
            if not row_slices:
                r.append(True)
                continue
            vstack_block = VerticalStackBlock(*matrices)
            output, dL_doutput = vstack_block.output.register_usage(context, context)
            _dL_output = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
            GpuMatrix.from_npa(_dL_output, 'float').copy(context, dL_doutput)
            vstack_block.fprop()
            vstack_block.bprop()
            for row_slice, dL_dmatrix in izip(row_slices, vstack_block.dL_dmatrices):
                a = dL_dmatrix.to_host()
                b = _dL_output[row_slice[0]:row_slice[1], :]
                if not np.allclose(a, b):
                    r.append(False)
                    break
            else:
                r.append(True)
        self.assertEqual(sum(r), self.N)