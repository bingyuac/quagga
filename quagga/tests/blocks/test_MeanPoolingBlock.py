import numpy as np
from unittest import TestCase
from quagga.context import Context
from quagga.matrix import GpuMatrix
from quagga.connector import Connector
from quagga.blocks import MeanPoolingBlock


class TestMeanPoolingBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 50

    @classmethod
    def get_random_array(cls, shape=None):
        if shape:
            a = 4 * cls.rng.rand(*shape) - 2
        else:
            nrows, ncols = cls.rng.randint(low=1, high=7000, size=2)
            a = 4 * cls.rng.rand(nrows, ncols) - 2
        return a.astype(dtype=np.float32)

    def test_fprop(self):
        r = []

        for i in xrange(self.N):
            a = self.get_random_array()
            a_gpu = Connector(GpuMatrix.from_npa(a, 'float'))
            vpooling_block = MeanPoolingBlock(a_gpu, axis=0)
            hpooling_block = MeanPoolingBlock(a_gpu, axis=1)

            vpooling_block.fprop()
            r.append(np.allclose(vpooling_block.output.to_host(),
                                 np.mean(a, axis=0, keepdims=True),
                                 atol=1e-6))
            hpooling_block.fprop()
            r.append(np.allclose(hpooling_block.output.to_host(),
                                 np.mean(a, axis=1, keepdims=True),
                                 atol=1e-6))

        self.assertEqual(sum(r), 2 * self.N)

    def test_bprop(self):
        r = []

        context = Context()
        for i in xrange(self.N):
            a = self.get_random_array()
            a_gpu = Connector(GpuMatrix.from_npa(a, 'float'), b_usage_context=context)
            vpooling_block = MeanPoolingBlock(a_gpu, axis=0)
            voutput, dL_dvoutput = vpooling_block.output.register_usage(context, context)
            _dL_voutput = self.get_random_array((dL_dvoutput.nrows, dL_dvoutput.ncols))
            GpuMatrix.from_npa(_dL_voutput, 'float').copy_to(context, dL_dvoutput)

            hpooling_block = MeanPoolingBlock(a_gpu, axis=1)
            houtput, dL_dhoutput = hpooling_block.output.register_usage(context, context)
            _dL_houtput = self.get_random_array((dL_dhoutput.nrows, dL_dhoutput.ncols))
            GpuMatrix.from_npa(_dL_houtput, 'float').copy_to(context, dL_dhoutput)

            vpooling_block.fprop()
            vpooling_block.bprop()
            dL_dmatrix = vpooling_block.dL_dmatrix.to_host()
            r.append(np.allclose(dL_dmatrix,
                                 np.repeat(_dL_voutput/a.shape[0], a.shape[0], 0),
                                 atol=1e-6))

            hpooling_block.fprop()
            hpooling_block.bprop()
            hpooling_block.dL_dmatrix.to_host()
            dL_dmatrix = hpooling_block.dL_dmatrix.to_host()
            r.append(np.allclose(dL_dmatrix,
                                 np.repeat(_dL_houtput/a.shape[1], a.shape[1], 1),
                                 atol=1e-6))

        self.assertEqual(sum(r), 2 * self.N)