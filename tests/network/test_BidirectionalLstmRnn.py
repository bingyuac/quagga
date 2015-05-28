import numpy as np
from unittest import TestCase
from matrix import initializers
from network import MatrixClass, BidirectionalLstmRnn


class TestBidirectionalLstmRnn(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.context = {}
        cls.N = 50

    # @classmethod
    # def get_random_array(cls, shape=None):
    #     if shape:
    #         a = 4 * cls.rng.rand(*shape) - 2
    #     else:
    #         nrows, ncols = cls.rng.random_integers(7000, size=2)
    #         a = 4 * cls.rng.rand(nrows, ncols) - 2
    #     return a.astype(dtype=np.float32)

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
        pass