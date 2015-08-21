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
    def get_random_array(cls, shape=None):
        if not shape:
            shape = cls.rng.random_integers(7000, size=2)
        a = cls.rng.normal(0, 1, shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        a = u if u.shape == shape else v
        return a.astype(dtype=np.float32)

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
            x_gpu.set_length(sequence_len)
            np_lstm_rnn_gpu = LstmRnn(W_init, R_init, x_gpu, learning=False)
            np_lstm_rnn_gpu.fprop()
            np_lstm_rnn_gpu.context.synchronize()
            h_gpu = np_lstm_rnn_gpu.h.to_host()

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            x_cpu = MatrixContainer([Connector(Matrix.from_npa(e)) for e in x])
            x_cpu.set_length(sequence_len)
            np_lstm_rnn_cpu = LstmRnn(W_init, R_init, x_cpu, learning=False)
            np_lstm_rnn_cpu.fprop()
            np_lstm_rnn_cpu.context.synchronize()
            h_cpu = np_lstm_rnn_cpu.h.to_host()

            for h_gpu, h_cpu in izip(h_gpu, h_cpu):
                if not np.allclose(h_gpu, h_cpu, rtol=1e-7, atol=1e-3):
                    r.append(False)
                    break
            else:
                r.append(True)
            del np_lstm_rnn_gpu
            del np_lstm_rnn_cpu
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
            batch_size = self.rng.random_integers(512)
            input_dim, hidden_dim = self.rng.random_integers(1500, size=2)
            x = [self.rng.rand(batch_size, input_dim).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]

            W_init = self.get_orthogonal_initializer(input_dim, hidden_dim)
            R_init = self.get_orthogonal_initializer(hidden_dim, hidden_dim)

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            context = Context()
            x_gpu = MatrixContainer([Connector(Matrix.from_npa(e), context, context) for e in x])
            x_gpu.set_length(sequence_len)
            np_lstm_rnn_gpu = LstmRnn(W_init, R_init, x_gpu)
            h, dL_dh = zip(*[h.register_usage(context, context) for h in np_lstm_rnn_gpu.h])
            np_lstm_rnn_gpu.fprop()
            for dL_dh in dL_dh:
                random_matrix = self.rng.rand(dL_dh.nrows, dL_dh.ncols)
                Matrix.from_npa(random_matrix, 'float').copy(context, dL_dh)
            np_lstm_rnn_gpu.bprop()
            context.synchronize()
            np_lstm_rnn_gpu.context.synchronize()
            dL_dW_gpu = np_lstm_rnn_gpu.dL_dW.to_host()
            dL_dR_gpu = np_lstm_rnn_gpu.dL_dR.to_host()
            dL_dx_gpu = [e.backward_matrix.to_host() for e in x_gpu]

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            context = Context()
            x_cpu = MatrixContainer([Connector(Matrix.from_npa(e), context, context) for e in x])
            x_cpu.set_length(sequence_len)
            np_lstm_rnn_cpu = LstmRnn(W_init, R_init, x_cpu)
            h, dL_dh = zip(*[h.register_usage(context, context) for h in np_lstm_rnn_cpu.h])
            np_lstm_rnn_cpu.fprop()
            for dL_dh in dL_dh:
                random_matrix = self.rng.rand(dL_dh.nrows, dL_dh.ncols)
                Matrix.from_npa(random_matrix, 'float').copy(context, dL_dh)
            np_lstm_rnn_cpu.bprop()
            context.synchronize()
            np_lstm_rnn_cpu.context.synchronize()
            dL_dW_cpu = np_lstm_rnn_cpu.dL_dW.to_host()
            dL_dR_cpu = np_lstm_rnn_cpu.dL_dR.to_host()
            dL_dx_cpu = [e.backward_matrix.to_host() for e in x_cpu]

            r.append(np.allclose(dL_dW_gpu, dL_dW_cpu, rtol=1e-7, atol=1e-3))
            r.append(np.allclose(dL_dR_gpu, dL_dR_cpu, rtol=1e-7, atol=1e-3))
            for dL_dx_gpu, dL_dx_cpu in izip(dL_dx_gpu, dL_dx_cpu):
                if not np.allclose(dL_dx_gpu, dL_dx_cpu, rtol=1e-7, atol=1e-3):
                    r.append(False)
                    break
            else:
                r.append(True)
            del np_lstm_rnn_gpu
            del np_lstm_rnn_cpu
            del x_gpu
            del x_cpu

        self.assertEqual(sum(r), self.N * 3)

    def test_finite_difference_x(self):
        quagga.processor_type = 'gpu'
        r = []
        n = 10

        for i in xrange(n):
            k = self.rng.random_integers(10)
            dim_x = self.rng.random_integers(50)
            dim_h = self.rng.random_integers(20)

            W_init = lambda: (self.rng.rand(dim_h, dim_x) * 0.1).astype(np.float32)
            W_init.nrows, W_init.ncols = dim_h, dim_x
            R_init = lambda: (self.rng.rand(dim_h, dim_h) * 0.1).astype(np.float32)
            R_init.nrows, R_init.ncols = dim_h, dim_h
            log_reg_init = lambda: (self.rng.rand(1, dim_h) * 0.1).astype(np.float32)

            x = Connector(Matrix.from_npa(self.rng.rand(dim_x, k), 'float'), b_usage_context=Context())
            true_labels = Connector(Matrix.from_npa(self.rng.choice(np.array([1, 0], dtype=np.float32), size=(1, k))))
            np_lstm_rnn = NpLstmRnnM(W_init, R_init, x)
            log_reg = LogisticRegressionCe(log_reg_init, np_lstm_rnn.h, true_labels)

            x.fprop()
            true_labels.fprop()
            np_lstm_rnn.fprop()
            log_reg.fprop()
            log_reg.bprop()
            np_lstm_rnn.bprop()

            dL_dx = x.backward_matrix.to_host()
            numerical_grad = np.zeros_like(dL_dx)
            cross_entropy = lambda l, p: -np.sum(l * np.log(p) + (1 - l) * np.log(1 - p))
            x_np = x.to_host()
            true_labels_np = true_labels.to_host()

            epsilon = 1e-2
            for i in xrange(x.nrows):
                for j in xrange(x.ncols):
                    x.__setitem__((i, j), x_np[i, j] + epsilon)
                    np_lstm_rnn.fprop()
                    log_reg.fprop()
                    probs = log_reg.probs.to_host()
                    plus_cost = cross_entropy(true_labels_np, probs)

                    x.__setitem__((i, j), x_np[i, j] - epsilon)
                    np_lstm_rnn.fprop()
                    log_reg.fprop()
                    probs = log_reg.probs.to_host()
                    minus_cost = cross_entropy(true_labels_np, probs)

                    numerical_grad[i, j] = (plus_cost - minus_cost) / (2 * epsilon)
                    x.__setitem__((i, j), x_np[i, j])

            r.append(np.allclose(dL_dx, numerical_grad, rtol=1e-7, atol=1e-4))

        self.assertEqual(sum(r), n)

    def test_finite_difference_w(self):
        quagga.processor_type = 'gpu'
        r = []
        n = 10

        for i in xrange(n):
            k = self.rng.random_integers(10)
            dim_x = self.rng.random_integers(50)
            dim_h = self.rng.random_integers(20)

            W_init = self.get_orthogonal_initializer(dim_h, dim_x)
            R_init = self.get_orthogonal_initializer(dim_h, dim_h)
            log_reg_init = lambda: (self.rng.rand(1, dim_h) * 0.1).astype(np.float32)

            x = Connector(Matrix.from_npa(self.rng.rand(dim_x, k), 'float'))
            true_labels = Connector(Matrix.from_npa(self.rng.choice(np.array([1, 0], dtype=np.float32), size=(1, k))))
            np_lstm_rnn = NpLstmRnnM(W_init, R_init, x)
            log_reg = LogisticRegressionCe(log_reg_init, np_lstm_rnn.h, true_labels)

            x.fprop()
            true_labels.fprop()
            np_lstm_rnn.fprop()
            log_reg.fprop()
            log_reg.bprop()
            np_lstm_rnn.bprop()

            dL_d = {'W': np_lstm_rnn.dL_dW.to_host(),
                    'R': np_lstm_rnn.dL_dR.to_host()}
            cross_entropy = lambda l, p: -np.sum(l * np.log(p) + (1 - l) * np.log(1 - p))
            true_labels_np = true_labels.to_host()
            for variable in ['W', 'R']:
                dL_dvariable = dL_d[variable]
                numerical_grad = np.zeros_like(dL_d[variable])
                variable = getattr(np_lstm_rnn, variable)
                variable_np = variable.to_host()
                epsilon = 1e-2
                for i in xrange(variable.nrows):
                    for j in xrange(variable.ncols):
                        variable.__setitem__((i, j), variable_np[i, j] + epsilon)
                        np_lstm_rnn.fprop()
                        log_reg.fprop()
                        probs = log_reg.probs.to_host()
                        plus_cost = cross_entropy(true_labels_np, probs)

                        variable.__setitem__((i, j), variable_np[i, j] - epsilon)
                        np_lstm_rnn.fprop()
                        log_reg.fprop()
                        probs = log_reg.probs.to_host()
                        minus_cost = cross_entropy(true_labels_np, probs)

                        numerical_grad[i, j] = (plus_cost - minus_cost) / (2 * epsilon)
                        variable.__setitem__((i, j), variable_np[i, j])

                r.append(np.allclose(dL_dvariable, numerical_grad, rtol=1e-7, atol=1e-4))

        self.assertEqual(sum(r), 2 * n)

    def test_theano_grad(self):
        class LstmLayer(object):
            def __init__(self, W_init, R_init):
                W_init = np.vstack((W_init(), W_init(), W_init(), W_init()))
                R_init = np.vstack((R_init(), R_init(), R_init(), R_init()))
                self.W = theano.shared(W_init, name='W_zifo')
                self.R = theano.shared(R_init, name='R_zifo')
                self.n = W_init.shape[0] / 4

            def get_output_expr(self, input_sequence):
                h0 = T.zeros((self.n, ), dtype=np.float32)
                c0 = T.zeros((self.n, ), dtype=np.float32)

                [_, h], _ = theano.scan(fn=self.__get_lstm_step_expr,
                                        sequences=input_sequence.T,
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
            k = self.rng.random_integers(10)
            dim_x = self.rng.random_integers(50)
            dim_h = self.rng.random_integers(20)

            W_init = self.get_orthogonal_initializer(dim_h, dim_x)
            R_init = self.get_orthogonal_initializer(dim_h, dim_h)
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
            np_lstm_rnn = NpLstmRnnM(W_init, R_init, x)
            log_reg = LogisticRegressionCe(log_reg_init, np_lstm_rnn.h, true_labels)

            x.fprop()
            true_labels.fprop()
            np_lstm_rnn.fprop()
            log_reg.fprop()
            log_reg.bprop()
            np_lstm_rnn.bprop()

            dL_d = {'W': np_lstm_rnn.dL_dW.to_host(),
                    'R': np_lstm_rnn.dL_dR.to_host(),
                    'x': x.backward_matrix.to_host()}
            theano_grad = dict(zip(['W', 'R', 'x'], get_theano_grads(x.to_host(), true_labels.to_host()[0])))

            for variable in ['W', 'R', 'x']:
                r.append(np.allclose(dL_d[variable], theano_grad[variable], rtol=1e-7, atol=1.e-7))

        self.assertEqual(sum(r), 3 * n)