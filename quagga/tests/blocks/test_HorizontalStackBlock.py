import numpy as np
from itertools import izip
from unittest import TestCase
from quagga.context import Context
from quagga.matrix import GpuMatrix
from quagga.connector import Connector
from quagga.blocks import HorizontalStackBlock


class TestHorizontalStackBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 50

    def test_fprop(self):
        r = []

        for i in xrange(self.N):
            matrices = []
            nrows = self.rng.random_integers(1, 5000)
            for _ in xrange(self.rng.random_integers(1, 10)):
                ncols = self.rng.random_integers(1, 5000)
                matrices.append(Connector(GpuMatrix.from_npa(np.random.rand(nrows, ncols), 'float')))
            hstack_block = HorizontalStackBlock(*matrices)
            hstack_block.fprop()
            a = hstack_block.output.to_host()
            b = np.hstack([matrix.to_host() for matrix in matrices])
            r.append(np.allclose(a, b))

        self.assertEqual(sum(r), self.N)

    def test_bprop(self):
        r = []

        context = Context()
        for i in xrange(self.N):
            matrices = []
            nrows = self.rng.random_integers(1, 3000)
            ncols = [0]
            col_slices = []
            for _ in xrange(self.rng.random_integers(1, 10)):
                _ncols = self.rng.random_integers(1, 2000)
                ncols.append(ncols[-1] + _ncols)
                if self.rng.choice([True, False]):
                    b_usage_context = context
                    col_slices.append((ncols[-2], ncols[-1]))
                else:
                    b_usage_context = None
                matrices.append(Connector(GpuMatrix.from_npa(np.random.rand(nrows, _ncols), 'float'), b_usage_context=b_usage_context))
            if not col_slices:
                r.append(True)
                continue
            hstack_block = HorizontalStackBlock(*matrices)
            output, dL_doutput = hstack_block.output.register_usage(context, context)
            _dL_output = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
            GpuMatrix.from_npa(_dL_output, 'float').copy(context, dL_doutput)
            hstack_block.fprop()
            hstack_block.bprop()
            for col_slice, dL_dmatrix in izip(col_slices, hstack_block.dL_dmatrices):
                a = dL_dmatrix.to_host()
                b = _dL_output[:, col_slice[0]:col_slice[1]]
                if not np.allclose(a, b):
                    r.append(False)
                    break
            else:
                r.append(True)
        self.assertEqual(sum(r), self.N)