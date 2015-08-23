import quagga
import theano
import numpy as np
from itertools import izip
from unittest import TestCase
from theano import tensor as T
from quagga.matrix import Matrix
from quagga.blocks import LstmRnn
from quagga.context import Context
from quagga.connector import Connector
from quagga.matrix import MatrixContainer
from quagga.blocks import LogisticRegressionCe


class TestLstmRnn(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 10

    @classmethod
    def get_orthogonal_initializer(cls, nrows, ncols):
        shape = (nrows, ncols)
        def initializer():
            a = cls.rng.normal(0.0, 1.0, shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == shape else v
            q = q.reshape(shape).astype(np.float32)
            return q
        initializer.nrows = shape[0]
        initializer.ncols = shape[1]
        return initializer

    def test_fprop(self):
        """
        compare `fprop` results for cpu and gpu backends
        """
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            batch_size = self.rng.random_integers(512)
            input_dim, hidden_dim = self.rng.random_integers(1500, size=2)
            x = [self.rng.rand(batch_size, input_dim).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]

            W_init = self.get_orthogonal_initializer(input_dim, hidden_dim)
            R_init = self.get_orthogonal_initializer(hidden_dim, hidden_dim)

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            x_gpu = MatrixContainer([Connector(Matrix.from_npa(e)) for e in x])
            lstm_rnn_gpu = LstmRnn(W_init, R_init, x_gpu, learning=False)
            x_gpu.set_length(sequence_len)
            lstm_rnn_gpu.fprop()
            lstm_rnn_gpu.context.synchronize()
            h_gpu = lstm_rnn_gpu.h.to_host()

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            x_cpu = MatrixContainer([Connector(Matrix.from_npa(e)) for e in x])
            lstm_rnn_cpu = LstmRnn(W_init, R_init, x_cpu, learning=False)
            x_cpu.set_length(sequence_len)
            lstm_rnn_cpu.fprop()
            lstm_rnn_cpu.context.synchronize()
            h_cpu = lstm_rnn_cpu.h.to_host()

            for h_gpu, h_cpu in izip(h_gpu, h_cpu):
                if not np.allclose(h_gpu, h_cpu, rtol=1e-7, atol=1e-3):
                    r.append(False)
                    break
            else:
                r.append(True)
            del lstm_rnn_gpu
            del lstm_rnn_cpu
            del x_gpu
            del x_cpu

        self.assertEqual(sum(r), self.N)

    def test_bprop(self):
        """
        compare `bprop` results for cpu and gpu backends
        """
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            batch_size = self.rng.random_integers(128)
            input_dim, hidden_dim = self.rng.random_integers(1500, size=2)
            x = [self.rng.rand(batch_size, input_dim).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]

            W_init = self.get_orthogonal_initializer(input_dim, hidden_dim)
            R_init = self.get_orthogonal_initializer(hidden_dim, hidden_dim)

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            context = Context()
            x_gpu = MatrixContainer([Connector(Matrix.from_npa(e), context, context) for e in x])
            lstm_rnn_gpu = LstmRnn(W_init, R_init, x_gpu)
            x_gpu.set_length(sequence_len)
            h, dL_dh = zip(*[h.register_usage(context, context) for h in lstm_rnn_gpu.h])
            lstm_rnn_gpu.fprop()
            for _, dL_dh in izip(h, dL_dh):
                random_matrix = self.rng.rand(dL_dh.nrows, dL_dh.ncols)
                Matrix.from_npa(random_matrix, 'float').copy(context, dL_dh)
            lstm_rnn_gpu.bprop()
            context.synchronize()
            lstm_rnn_gpu.context.synchronize()
            dL_dW_gpu = lstm_rnn_gpu.dL_dW.to_host()
            dL_dR_gpu = lstm_rnn_gpu.dL_dR.to_host()
            dL_dx_gpu = [e.backward_matrix.to_host() for e in x_gpu]

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            context = Context()
            x_cpu = MatrixContainer([Connector(Matrix.from_npa(e), context, context) for e in x])
            lstm_rnn_cpu = LstmRnn(W_init, R_init, x_cpu)
            x_cpu.set_length(sequence_len)
            h, dL_dh = zip(*[h.register_usage(context, context) for h in lstm_rnn_cpu.h])
            lstm_rnn_cpu.fprop()
            for _, dL_dh in izip(h, dL_dh):
                random_matrix = self.rng.rand(dL_dh.nrows, dL_dh.ncols)
                Matrix.from_npa(random_matrix, 'float').copy(context, dL_dh)
            lstm_rnn_cpu.bprop()
            context.synchronize()
            lstm_rnn_cpu.context.synchronize()
            dL_dW_cpu = lstm_rnn_cpu.dL_dW.to_host()
            dL_dR_cpu = lstm_rnn_cpu.dL_dR.to_host()
            dL_dx_cpu = [e.backward_matrix.to_host() for e in x_cpu]

            r.append(np.allclose(dL_dW_gpu, dL_dW_cpu, rtol=1e-7, atol=1e-3))
            r.append(np.allclose(dL_dR_gpu, dL_dR_cpu, rtol=1e-7, atol=1e-3))
            for dL_dx_gpu, dL_dx_cpu in izip(dL_dx_gpu, dL_dx_cpu):
                if not np.allclose(dL_dx_gpu, dL_dx_cpu, rtol=1e-7, atol=1e-3):
                    r.append(False)
                    break
            else:
                r.append(True)
            del lstm_rnn_gpu
            del lstm_rnn_cpu
            del x_gpu
            del x_cpu

        self.assertEqual(sum(r), self.N * 3)

    def test_theano_grad(self):
        class LstmLayer(object):
            def __init__(self, W_init, R_init):
                W_init = np.vstack((W_init(), W_init(), W_init(), W_init()))
                R_init = np.vstack((R_init(), R_init(), R_init(), R_init()))
                self.W = theano.shared(W_init, name='W_zifo')
                self.R = theano.shared(R_init, name='R_zifo')
                self.n = W_init.shape[0] / 4

            def get_output_expr(self, input_sequence):
                h0 = T.zeros((batch_size, self.n), dtype=np.float32)
                c0 = T.zeros((batch_size, self.n), dtype=np.float32)

                [_, h], _ = theano.scan(fn=self.__get_lstm_step_expr,
                                        sequences=input_sequence,
                                        outputs_info=[c0, h0])
                return h.T

            def __get_lstm_step_expr(self, x_t, c_tm1, h_tm1):
                sigm = T.nnet.sigmoid
                tanh = T.tanh
                dot = theano.dot

                zifo_t = dot(self.W, x_t) + dot(self.R, h_tm1)
                z_t = tanh(zifo_t[0*self.n:1*self.n])
                i_t = sigm(zifo_t[1*self.n:2*self.n])
                f_t = sigm(zifo_t[2*self.n:3*self.n])
                o_t = sigm(zifo_t[3*self.n:4*self.n])

                c_t = i_t * z_t + f_t * c_tm1
                h_t = o_t * tanh(c_t)
                return c_t, h_t

        class LogisticRegression(object):
            def __init__(self, W_init):
                self.W = theano.shared(value=W_init(), name='W')

            def get_output_expr(self, input_expr):
                return T.nnet.sigmoid(T.dot(self.W, input_expr))

        quagga.processor_type = 'gpu'
        r = []
        n = 10

        for i in xrange(n):
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            batch_size = self.rng.random_integers(128)
            input_dim, hidden_dim = self.rng.random_integers(1500, size=2)
            x = [self.rng.rand(batch_size, input_dim).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]

            W_init = self.get_orthogonal_initializer(input_dim, hidden_dim)
            R_init = self.get_orthogonal_initializer(hidden_dim, hidden_dim)
            log_reg_init = lambda: (self.rng.rand(1, dim_h) * 0.1).astype(np.float32)

            x = Connector(Matrix.from_npa(self.rng.rand(dim_x, k), 'float'), b_usage_context=Context())
            true_labels = Connector(Matrix.from_npa(self.rng.choice(np.array([1, 0], dtype=np.float32), size=(1, k))))

            # Theano model
            state = self.rng.get_state()
            th_x = T.fmatrix('x')
            th_true_labels = T.fvector('true_labels')
            lstm_layer = LstmLayer(W_init, R_init)
            lr_layer = LogisticRegression(log_reg_init)

            output = lr_layer.get_output_expr(lstm_layer.get_output_expr(th_x))
            loss = T.sum(T.nnet.binary_crossentropy(output, th_true_labels))
            grad_W, grad_R, grad_x = T.grad(loss, wrt=[lstm_layer.W, lstm_layer.R, th_x])
            get_theano_grads = theano.function([th_x, th_true_labels], [grad_W, grad_R, grad_x])
            self.rng.set_state(state)

            # quagga model
            lstm_rnn = NpLstmRnnM(W_init, R_init, x)
            log_reg = LogisticRegressionCe(log_reg_init, lstm_rnn.h, true_labels)

            x.fprop()
            true_labels.fprop()
            lstm_rnn.fprop()
            log_reg.fprop()
            log_reg.bprop()
            lstm_rnn.bprop()

            dL_d = {'W': lstm_rnn.dL_dW.to_host(),
                    'R': lstm_rnn.dL_dR.to_host(),
                    'x': x.backward_matrix.to_host()}
            theano_grad = dict(zip(['W', 'R', 'x'], get_theano_grads(x.to_host(), true_labels.to_host()[0])))

            for variable in ['W', 'R', 'x']:
                r.append(np.allclose(dL_d[variable], theano_grad[variable], rtol=1e-7, atol=1.e-7))

        self.assertEqual(sum(r), 3 * n)