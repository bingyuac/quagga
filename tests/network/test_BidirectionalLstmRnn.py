import ctypes
import numpy as np
from unittest import TestCase
from matrix import initializers
from network import MatrixClass, BidirectionalLstmRnn


class TestBidirectionalLstmRnn(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.context = {}
        cls.N = 5

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
        max_input_sequence_len = 120
        for i in xrange(self.N):
            nrows, ncols = self.rng.random_integers(1000, size=2)
            nrows, ncols = 10, 20
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
                print lstm_rnn.forward_propagation(input_sequence)
                dL_dW[p_type], dL_dR[p_type], dL_dp[p_type], dL_dw_hy[p_type], dL_dx[p_type] = lstm_rnn.backward_propagation(input_sequence, sequence_grammaticality)
                initializers.rng.set_state(random_state)
            r.append(np.allclose(dL_dW['cpu'], dL_dW['gpu'], atol=1e-4))
            r.append(np.allclose(dL_dR['cpu'], dL_dR['gpu'], atol=1e-4))
            r.append(np.allclose(dL_dp['cpu'], dL_dp['gpu'], atol=1e-4))
            r.append(np.allclose(dL_dw_hy['cpu'], dL_dw_hy['gpu'], atol=1e-4))
            r.append(np.allclose(dL_dx['cpu'], dL_dx['gpu'], atol=1e-4))
        self.assertEqual(sum(r), self.N * 5)