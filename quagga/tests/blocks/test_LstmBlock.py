import quagga
import numpy as np
from unittest import TestCase
from quagga.matrix import Matrix, MatrixContext
from quagga.blocks import LstmBlock, MarginalLstmBlock


class TestLstmBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.context = {}
        for p_type in quagga.get_processors_types():
            for gate in 'zifco':
                cls.context[p_type, gate] = MatrixContext()
        cls.N = 50

    @classmethod
    def get_random_array(cls, shape=None):
        if shape:
            a = 4 * cls.rng.rand(*shape) - 2
        else:
            nrows, ncols = cls.rng.random_integers(7000, size=2)
            a = 4 * cls.rng.rand(nrows, ncols) - 2
        return a.astype(dtype=np.float32)

    def test_forward_propagation(self):
        r = []
        for i in xrange(self.N):
            nrows, ncols = self.rng.random_integers(7000, size=2)
            W, R, p, pre = {}, {}, {}, {}
            for gate in 'zifo':
                _W = self.get_random_array((nrows, ncols))
                _R = self.get_random_array((nrows, nrows))
                if gate != 'z':
                    _p = self.get_random_array((nrows, 1))
                _pre = self.get_random_array((nrows, 1))
                for p_type in quagga.get_processors_types():
                    W[p_type, gate] = Matrix.from_npa(_W)
                    R[p_type, gate] = Matrix.from_npa(_R)
                    if gate != 'z':
                        p[p_type, gate] = Matrix.from_npa(_p)
                    pre[p_type, gate] = Matrix.from_npa(_pre)

            c, h = {}, {}
            for p_type in quagga.get_processors_types():
                cell = LstmBlock(W[p_type, 'z'], R[p_type, 'z'],
                                 W[p_type, 'i'], R[p_type, 'i'], p[p_type, 'i'],
                                 W[p_type, 'f'], R[p_type, 'f'], p[p_type, 'f'],
                                 W[p_type, 'o'], R[p_type, 'o'], p[p_type, 'o'],
                                 Matrix.empty(*_p.shape),
                                 Matrix.empty(*_p.shape),
                                 None, None, None, None,
                                 self.context[p_type, 'z'],
                                 self.context[p_type, 'i'],
                                 self.context[p_type, 'f'],
                                 self.context[p_type, 'c'],
                                 self.context[p_type, 'o'],
                                 MarginalLstmBlock(_p.shape[0]))
                cell.set_testing_mode()
                cell.fprop(pre[p_type, 'z'], pre[p_type, 'i'], pre[p_type, 'f'], pre[p_type, 'o'])
                cell.o_context.synchronize()
                c[p_type] = cell.c
                h[p_type] = cell.h

            r.append(np.allclose(c['cpu'].to_host(), c['gpu'].to_host()))
            r.append(np.allclose(h['cpu'].to_host(), h['gpu'].to_host()))

        self.assertEqual(sum(r), self.N * 2)

    def test_backward_propagation(self):
        r = []
        for i in xrange(self.N):
            nrows, ncols = self.rng.random_integers(7000, size=2)
            W, R, p, pre, dL_dpre_tp1, dL_dc_tp1, f, dL_dh = {}, {}, {}, {}, {}, {}, {}, {}
            for gate in 'zifo':
                _W = self.get_random_array((nrows, ncols))
                _R = self.get_random_array((nrows, nrows))
                if gate != 'z':
                    _p = self.get_random_array((nrows, 1))
                else:
                    _dL_dc = self.get_random_array((nrows, 1))
                    _f = self.get_random_array((nrows, 1))
                    _dL_dh = self.get_random_array((nrows, 1))
                _pre = self.get_random_array((nrows, 1))
                _dL_dpre = self.get_random_array((nrows, 1))

                for p_type in quagga.get_processors_types():
                    W[p_type, gate] = Matrix.from_npa(_W)
                    R[p_type, gate] = Matrix.from_npa(_R)
                    if gate != 'z':
                        p[p_type, gate] = Matrix.from_npa(_p)
                    else:
                        dL_dc_tp1[p_type] = Matrix.from_npa(_dL_dc)
                        f[p_type] = Matrix.from_npa(_f)
                        dL_dh[p_type] = Matrix.from_npa(_dL_dh)
                    pre[p_type, gate] = Matrix.from_npa(_pre)
                    dL_dpre_tp1[p_type, gate] = Matrix.from_npa(_dL_dpre)

            dL_dpre, dL_dc = {}, {}
            is_cell_last = self.rng.randint(2)
            for p_type in quagga.get_processors_types():
                prev_cell = MarginalLstmBlock(_p.shape[0])
                next_cell = MarginalLstmBlock(_p.shape[0])
                next_cell.dL_dpre_z = dL_dpre_tp1[p_type, 'z']
                next_cell.dL_dpre_i = dL_dpre_tp1[p_type, 'i']
                next_cell.dL_dpre_f = dL_dpre_tp1[p_type, 'f']
                next_cell.dL_dpre_o = dL_dpre_tp1[p_type, 'o']
                next_cell.dL_dc = dL_dc_tp1[p_type]
                next_cell.f = f[p_type]

                cell = LstmBlock(W[p_type, 'z'], R[p_type, 'z'],
                                 W[p_type, 'i'], R[p_type, 'i'], p[p_type, 'i'],
                                 W[p_type, 'f'], R[p_type, 'f'], p[p_type, 'f'],
                                 W[p_type, 'o'], R[p_type, 'o'], p[p_type, 'o'],
                                 Matrix.empty(*_p.shape),
                                 Matrix.empty(*_p.shape),
                                 Matrix.empty(*_p.shape),
                                 Matrix.empty(*_p.shape),
                                 Matrix.empty(*_p.shape),
                                 Matrix.empty(*_p.shape),
                                 self.context[p_type, 'z'],
                                 self.context[p_type, 'i'],
                                 self.context[p_type, 'f'],
                                 self.context[p_type, 'c'],
                                 self.context[p_type, 'o'],
                                 prev_cell, next_cell)
                cell.set_training_mode()
                cell.fprop(pre[p_type, 'z'], pre[p_type, 'i'], pre[p_type, 'f'], pre[p_type, 'o'])
                cell.bprop(dL_dh[p_type] if is_cell_last else None)

                dL_dpre[p_type, 'z'] = cell.dL_dpre_z
                dL_dpre[p_type, 'i'] = cell.dL_dpre_i
                dL_dpre[p_type, 'f'] = cell.dL_dpre_f
                dL_dpre[p_type, 'o'] = cell.dL_dpre_o
                dL_dc[p_type] = cell.dL_dc

            r.append(np.allclose(dL_dpre['cpu', 'z'].to_host(), dL_dpre['gpu', 'z'].to_host(), atol=1e-3))
            r.append(np.allclose(dL_dpre['cpu', 'i'].to_host(), dL_dpre['gpu', 'i'].to_host(), atol=1e-3))
            r.append(np.allclose(dL_dpre['cpu', 'f'].to_host(), dL_dpre['gpu', 'f'].to_host(), atol=1e-3))
            r.append(np.allclose(dL_dpre['cpu', 'o'].to_host(), dL_dpre['gpu', 'o'].to_host(), atol=1e-3))
            r.append(np.allclose(dL_dc['cpu'].to_host(), dL_dc['gpu'].to_host(), atol=1e-3))

        self.assertEqual(sum(r), self.N * 5)