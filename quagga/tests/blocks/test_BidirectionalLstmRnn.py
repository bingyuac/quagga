import ctypes
import numpy as np
from unittest import TestCase
from quagga.matrix import initializers, Matrix


class TestBidirectionalLstmRnn(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.context = {}
        cls.N = 50

    def test_forward_propagation(self):
        r = []
        max_input_sequence_len = 120
        for i in xrange(self.N):
            nrows, ncols = self.rng.random_integers(1000, size=2)
            W_init = initializers.Orthogonal(nrows, ncols)
            R_init = initializers.Orthogonal(nrows, nrows)
            p_init = initializers.Orthogonal(nrows, 1)
            logistic_init = initializers.Orthogonal(nrows, 1)
            random_state = initializers.rng.get_state()
            prediction = {}
            for p_type in ['cpu', 'gpu']:
                lstm_rnn = BidirectionalLstmRnn(p_type, max_input_sequence_len, W_init, R_init, p_init, logistic_init)
                lstm_rnn.set_testing_mode()
                input_sequence_len = initializers.rng.random_integers(max_input_sequence_len)
                _input_sequence = initializers.Orthogonal(ncols, input_sequence_len)()
                input_sequence = MatrixClass[p_type].from_npa(_input_sequence)
                prediction[p_type] = lstm_rnn.forward_propagation(input_sequence)
                initializers.rng.set_state(random_state)
            r.append(np.allclose(prediction['cpu'], prediction['gpu'], atol=1e-4))
        self.assertEqual(sum(r), self.N)

    def test_backward_propagation(self):
        r = []
        max_input_sequence_len = 20
        for i in xrange(self.N):
            nrows, ncols = self.rng.random_integers(1000, size=2)
            print nrows, ncols
            W_init = initializers.Orthogonal(nrows, ncols)
            R_init = initializers.Orthogonal(nrows, nrows)
            p_init = initializers.Orthogonal(nrows, 1)
            logistic_init = initializers.Orthogonal(nrows, 1)
            random_state = initializers.rng.get_state()
            dL_dW, dL_dR, dL_dp, dL_dw_hy, dL_dx = {}, {}, {}, {}, {}
            for p_type in ['cpu', 'gpu']:
                lstm_rnn = BidirectionalLstmRnn(p_type, max_input_sequence_len, W_init, R_init, p_init, logistic_init)
                lstm_rnn.set_training_mode()
                input_sequence_len = initializers.rng.random_integers(max_input_sequence_len)
                _input_sequence = initializers.Orthogonal(ncols, input_sequence_len)()
                input_sequence = MatrixClass[p_type].from_npa(_input_sequence)
                sequence_grammaticality = initializers.rng.randint(2)
                sequence_grammaticality = ctypes.c_float(sequence_grammaticality)
                error, dL_dW[p_type], dL_dR[p_type], dL_dp[p_type], dL_dw_hy[p_type], dL_dx[p_type] = lstm_rnn.backward_propagation(input_sequence, sequence_grammaticality)
                initializers.rng.set_state(random_state)
                lstm_rnn.synchronize()
            print '===='
            for d in ['forward', 'backward']:
                r.append(np.allclose(dL_dw_hy['cpu'][d].to_host(), dL_dw_hy['gpu'][d].to_host(), atol=1e-3))
                for gate in 'zifo':
                    r.append(np.allclose(dL_dW['cpu'][gate, d].to_host(),
                                         dL_dW['gpu'][gate, d].to_host(), atol=1e-3))
                    r.append(np.allclose(dL_dR['cpu'][gate, d].to_host(),
                                         dL_dR['gpu'][gate, d].to_host(), atol=1e-3))
                    if gate != 'z':
                        r.append(np.allclose(dL_dp['cpu'][gate, d].to_host(),
                                             dL_dp['gpu'][gate, d].to_host(), atol=1e-3))
            r.append(np.allclose(dL_dx['cpu'].to_host(), dL_dx['gpu'].to_host(), atol=1e-3))

        self.assertEqual(sum(r), self.N * 25)