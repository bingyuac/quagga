import quagga
import numpy as np
from unittest import TestCase
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import NpLstmRnn
from quagga.connector import Connector


class TestNpLstmRnn(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 10

    @classmethod
    def get_random_array(cls, shape=None):
        if not shape:
            shape = cls.rng.random_integers(7000, size=2)
        a = cls.rng.normal(0, 1, shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        a = u if u.shape == shape else v
        return a.astype(dtype=np.float32)

    def test_fprop(self):
        r = []
        for i in xrange(self.N):
            nrows, ncols = self.rng.random_integers(2000, size=2)
            max_input_sequence_len = self.rng.random_integers(1000)
            x = 4 * self.rng.rand(ncols, max_input_sequence_len).astype(dtype=np.float32) - 2

            W = []
            for k in xrange(4):
                W.append(self.get_random_array((nrows, ncols)))
            def W_init():
                W_init.wk = (W_init.wk + 1) % 4
                return W[W_init.wk]
            W_init.wk = -1
            W_init.nrows = nrows
            W_init.ncols = ncols

            R = []
            for k in xrange(4):
                R.append(self.get_random_array((nrows, nrows)))
            def R_init():
                R_init.rk = (R_init.rk + 1) % 4
                return R[R_init.rk]
            R_init.rk = -1
            R_init.nrows = nrows
            R_init.ncols = nrows

            quagga.processor_type = 'gpu'
            x_gpu = Connector(Matrix.from_npa(x))
            np_lstm_rnn_gpu = NpLstmRnn(W_init, R_init, x_gpu, learning=False)
            np_lstm_rnn_gpu.fprop()
            np_lstm_rnn_gpu.context.synchronize()
            h_gpu = np_lstm_rnn_gpu.h.to_host()

            quagga.processor_type = 'cpu'
            x_cpu = Connector(Matrix.from_npa(x))
            np_lstm_rnn_cpu = NpLstmRnn(W_init, R_init, x_cpu, learning=False)
            np_lstm_rnn_cpu.fprop()
            np_lstm_rnn_cpu.context.synchronize()
            h_cpu = np_lstm_rnn_cpu.h.to_host()

            r.append(np.allclose(h_gpu, h_cpu, atol=1e-3))

        self.assertEqual(sum(r), self.N)