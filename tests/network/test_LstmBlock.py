import numpy as np
from unittest import TestCase
from network import LstmBlock, MarginalLstmBlock, MatrixClass, MatrixContextClass


class TestLstmBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.context = {}
        for p_type in ['cpu', 'gpu']:
            for gate in ['z', 'i', 'f', 'c', 'o']:
                cls.context[p_type, gate] = MatrixContextClass[p_type]()
        cls.N = 50

    @classmethod
    def get_random_array(cls, shape=None):
        if shape:
            a = 4 * cls.rng.rand(*shape) - 2
        else:
            nrows, ncols = cls.rng.randint(low=1, high=7000, size=2)
            a = 4 * cls.rng.rand(nrows, ncols) - 2
        return a.astype(dtype=np.float32)

    def test_forward_propagation(self):
        r = []
        for i in xrange(self.N):
            W, R, p, pre = {}, {}, {}, {}
            _W = self.get_random_array()
            _R = self.get_random_array((_W.shape[0], _W.shape[0]))
            _pre = self.get_random_array((_W.shape[0], 1))
            for p_type in ['cpu', 'gpu']:
                W[p_type, 'z'] = MatrixClass[p_type].from_npa(_W)
                R[p_type, 'z'] = MatrixClass[p_type].from_npa(_R)
                pre[p_type, 'z'] = MatrixClass[p_type].from_npa(_pre)

            for gate in ['i', 'f', 'o']:
                _W = self.get_random_array(_W.shape)
                _R = self.get_random_array(_R.shape)
                _p = self.get_random_array(_pre.shape)
                _pre = self.get_random_array(_pre.shape)
                for p_type in ['cpu', 'gpu']:
                    W[p_type, gate] = MatrixClass[p_type].from_npa(_W)
                    R[p_type, gate] = MatrixClass[p_type].from_npa(_R)
                    p[p_type, gate] = MatrixClass[p_type].from_npa(_p)
                    pre[p_type, gate] = MatrixClass[p_type].from_npa(_pre)

            c, h = {}, {}
            for p_type in ['cpu', 'gpu']:
                cell = LstmBlock(p_type,
                                 W[p_type, 'z'], R[p_type, 'z'],
                                 W[p_type, 'i'], R[p_type, 'i'], p[p_type, 'i'],
                                 W[p_type, 'f'], R[p_type, 'f'], p[p_type, 'f'],
                                 W[p_type, 'o'], R[p_type, 'o'], p[p_type, 'o'],
                                 MatrixClass[p_type].empty(*_p.shape),
                                 MatrixClass[p_type].empty(*_p.shape),
                                 None, None, None, None,
                                 self.context[p_type, 'z'],
                                 self.context[p_type, 'i'],
                                 self.context[p_type, 'f'],
                                 self.context[p_type, 'c'],
                                 self.context[p_type, 'o'])
                cell.prev_cell = MarginalLstmBlock(p_type, _p.shape[0])
                cell.set_testing_mode()
                cell.forward_propagation(pre[p_type, 'z'], pre[p_type, 'i'], pre[p_type, 'f'], pre[p_type, 'o'])
                cell.o_context.synchronize()
                c[p_type] = cell.c
                h[p_type] = cell.h

            r.append(np.allclose(c['cpu'].to_host(), c['gpu'].to_host()))
            r.append(np.allclose(h['cpu'].to_host(), h['gpu'].to_host()))

        self.assertEqual(sum(r), self.N * 2)