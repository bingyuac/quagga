import quagga
import theano
import numpy as np
from itertools import izip
from unittest import TestCase
from quagga.cuda import cudart
from theano import tensor as T
from quagga.matrix import Matrix
from quagga.blocks import LstmRnn
from quagga.context import Context
from quagga.blocks import DotBlock
from quagga.matrix import MatrixList
from quagga.connector import Connector
from quagga.blocks import SigmoidCeBlock
from quagga.blocks import SequentialMeanPoolingBlock


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
            batch_size = self.rng.random_integers(256)
            input_dim, hidden_dim = self.rng.random_integers(1500, size=2)
            x = [self.rng.rand(batch_size, input_dim).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]

            W_init = self.get_orthogonal_initializer(input_dim, hidden_dim)
            R_init = self.get_orthogonal_initializer(hidden_dim, hidden_dim)

            state = self.rng.get_state()
            for reverse in [False, True]:
                self.rng.set_state(state)
                quagga.processor_type = 'gpu'
                x_gpu = MatrixList([Connector(Matrix.from_npa(e)) for e in x])
                lstm_rnn_gpu = LstmRnn(W_init, R_init, x_gpu, reverse, learning=False)
                x_gpu.set_length(sequence_len)
                lstm_rnn_gpu.fprop()
                lstm_rnn_gpu.context.synchronize()
                h_gpu = lstm_rnn_gpu.h.to_host()

                self.rng.set_state(state)
                quagga.processor_type = 'cpu'
                x_cpu = MatrixList([Connector(Matrix.from_npa(e)) for e in x])
                lstm_rnn_cpu = LstmRnn(W_init, R_init, x_cpu, reverse, learning=False)
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
                del x_gpu

        self.assertEqual(sum(r), self.N * 2)

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
            for reverse in [False, True]:
                self.rng.set_state(state)
                quagga.processor_type = 'gpu'
                context = Context()
                x_gpu = MatrixList([Connector(Matrix.from_npa(e), context, context) for e in x])
                lstm_rnn_gpu = LstmRnn(W_init, R_init, x_gpu, reverse)
                x_gpu.set_length(sequence_len)
                h, dL_dh = zip(*[h.register_usage(context, context) for h in lstm_rnn_gpu.h])
                lstm_rnn_gpu.fprop()
                for _, dL_dh in izip(h, dL_dh):
                    random_matrix = self.rng.rand(dL_dh.nrows, dL_dh.ncols)
                    Matrix.from_npa(random_matrix, 'float').copy(context, dL_dh)
                lstm_rnn_gpu.bprop()
                lstm_rnn_gpu.context.synchronize()
                dL_dW_gpu = lstm_rnn_gpu.dL_dW.to_host()
                dL_dR_gpu = lstm_rnn_gpu.dL_dR.to_host()
                dL_dx_gpu = [e.backward_matrix.to_host() for e in x_gpu]

                self.rng.set_state(state)
                quagga.processor_type = 'cpu'
                context = Context()
                x_cpu = MatrixList([Connector(Matrix.from_npa(e), context, context) for e in x])
                lstm_rnn_cpu = LstmRnn(W_init, R_init, x_cpu, reverse)
                x_cpu.set_length(sequence_len)
                h, dL_dh = zip(*[h.register_usage(context, context) for h in lstm_rnn_cpu.h])
                lstm_rnn_cpu.fprop()
                for _, dL_dh in izip(h, dL_dh):
                    random_matrix = self.rng.rand(dL_dh.nrows, dL_dh.ncols)
                    Matrix.from_npa(random_matrix, 'float').copy(context, dL_dh)
                lstm_rnn_cpu.bprop()
                lstm_rnn_cpu.context.synchronize()
                dL_dW_cpu = lstm_rnn_cpu.dL_dW.to_host()
                dL_dR_cpu = lstm_rnn_cpu.dL_dR.to_host()
                dL_dx_cpu = [e.backward_matrix.to_host() for e in x_cpu]

                r.append(np.allclose(dL_dW_gpu, dL_dW_cpu, rtol=1e-7, atol=2e-3))
                r.append(np.allclose(dL_dR_gpu, dL_dR_cpu, rtol=1e-7, atol=2e-3))
                for dL_dx_gpu, dL_dx_cpu in izip(dL_dx_gpu, dL_dx_cpu):
                    if not np.allclose(dL_dx_gpu, dL_dx_cpu, rtol=1e-7, atol=2e-3):
                        r.append(False)
                        break
                else:
                    r.append(True)
                del lstm_rnn_gpu
                del x_gpu

        self.assertEqual(sum(r), self.N * 6)

    def test_theano_fprop(self):
        class LstmLayer(object):
            def __init__(self, W_init, R_init, reverse):
                W_init = np.hstack((W_init(), W_init(), W_init(), W_init()))
                R_init = np.hstack((R_init(), R_init(), R_init(), R_init()))
                self.W = theano.shared(W_init, name='W_zifo')
                self.R = theano.shared(R_init, name='R_zifo')
                self.n = W_init.shape[1] / 4
                self.reverse = reverse

            def get_output_expr(self, input_sequence):
                h0 = T.zeros((batch_size, self.n), dtype=np.float32)
                c0 = T.zeros((batch_size, self.n), dtype=np.float32)
                input_sequence = input_sequence.transpose(2, 0, 1)
                if reverse:
                    input_sequence = input_sequence[::-1]
                [_, h], _ = theano.scan(fn=self.__get_lstm_step_expr,
                                        sequences=input_sequence,
                                        outputs_info=[c0, h0])
                return h[::-1] if reverse else h

            def __get_lstm_step_expr(self, x_t, c_tm1, h_tm1):
                sigm = T.nnet.sigmoid
                tanh = T.tanh
                dot = theano.dot

                zifo_t = dot(x_t, self.W) + dot(h_tm1, self.R)
                z_t = tanh(zifo_t[:, 0*self.n:1*self.n])
                i_t = sigm(zifo_t[:, 1*self.n:2*self.n])
                f_t = sigm(zifo_t[:, 2*self.n:3*self.n])
                o_t = sigm(zifo_t[:, 3*self.n:4*self.n])

                c_t = i_t * z_t + f_t * c_tm1
                h_t = o_t * tanh(c_t)
                return c_t, h_t

        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            batch_size = self.rng.random_integers(256)
            input_dim, hidden_dim = self.rng.random_integers(1500, size=2)
            x = [self.rng.rand(batch_size, input_dim).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]

            W_init = self.get_orthogonal_initializer(input_dim, hidden_dim)
            R_init = self.get_orthogonal_initializer(hidden_dim, hidden_dim)

            state = self.rng.get_state()
            for reverse in [False, True]:
                self.rng.set_state(state)
                th_x = T.ftensor3()
                lstm_layer = LstmLayer(W_init, R_init, reverse)
                th_h = theano.function([th_x], lstm_layer.get_output_expr(th_x))
                th_h = th_h(np.dstack(x)[..., :sequence_len])

                self.rng.set_state(state)
                q_x = MatrixList([Connector(Matrix.from_npa(e)) for e in x])
                lstm_rnn_gpu = LstmRnn(W_init, R_init, q_x, reverse, learning=False)
                q_x.set_length(sequence_len)
                for e in q_x:
                    e.fprop()
                lstm_rnn_gpu.fprop()
                cudart.cuda_device_synchronize()
                q_h = lstm_rnn_gpu.h.to_host()

                for i in xrange(th_h.shape[0]):
                    if not np.allclose(q_h[i], th_h[i]):
                        r.append(False)
                        break
                else:
                    r.append(True)
                del lstm_rnn_gpu
                del q_x

        self.assertEqual(sum(r), self.N * 2)

    def test_theano_grad(self):
        class LstmLayer(object):
            def __init__(self, W_init, R_init, reverse):
                W_init = np.hstack((W_init(), W_init(), W_init(), W_init()))
                R_init = np.hstack((R_init(), R_init(), R_init(), R_init()))
                self.W = theano.shared(W_init, name='W_zifo')
                self.R = theano.shared(R_init, name='R_zifo')
                self.n = W_init.shape[1] / 4
                self.reverse = reverse

            def get_output_expr(self, input_sequence):
                h0 = T.zeros((batch_size, self.n), dtype=np.float32)
                c0 = T.zeros((batch_size, self.n), dtype=np.float32)
                input_sequence = input_sequence.transpose(2, 0, 1)
                if reverse:
                    input_sequence = input_sequence[::-1]
                [_, h], _ = theano.scan(fn=self.__get_lstm_step_expr,
                                        sequences=input_sequence,
                                        outputs_info=[c0, h0])
                return h[::-1] if reverse else h

            def __get_lstm_step_expr(self, x_t, c_tm1, h_tm1):
                sigm = T.nnet.sigmoid
                tanh = T.tanh
                dot = theano.dot

                zifo_t = dot(x_t, self.W) + dot(h_tm1, self.R)
                z_t = tanh(zifo_t[:, 0*self.n:1*self.n])
                i_t = sigm(zifo_t[:, 1*self.n:2*self.n])
                f_t = sigm(zifo_t[:, 2*self.n:3*self.n])
                o_t = sigm(zifo_t[:, 3*self.n:4*self.n])

                c_t = i_t * z_t + f_t * c_tm1
                h_t = o_t * tanh(c_t)
                return c_t, h_t

        class SequentialMeanPoolingLayer(object):
            def get_output_expr(self, input_sequence):
                return T.mean(input_sequence, axis=0)

        class LogisticRegressionLayer(object):
            def __init__(self, W_init, b_init):
                self.W = theano.shared(value=W_init())
                self.b = theano.shared(value=b_init())

            def get_output_expr(self, input_expr):
                return T.nnet.sigmoid(T.dot(input_expr, self.W) + self.b)

        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            batch_size = self.rng.random_integers(128)
            input_dim, hidden_dim = self.rng.random_integers(1500, size=2)
            x = [self.rng.rand(batch_size, input_dim).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]
            true_labels = self.rng.randint(2, size=(batch_size, 1)).astype(dtype=np.float32)

            W_init = self.get_orthogonal_initializer(input_dim, hidden_dim)
            R_init = self.get_orthogonal_initializer(hidden_dim, hidden_dim)
            lr_W_init = self.get_orthogonal_initializer(hidden_dim, 1)
            lr_b_init = lambda: self.rng.rand(1, 1).astype(dtype=np.float32)

            state = self.rng.get_state()
            for reverse in [False, True]:
                # Theano model
                self.rng.set_state(state)
                th_x = T.ftensor3()
                th_true_labels = T.fmatrix()
                lstm_layer = LstmLayer(W_init, R_init, reverse)
                smp_layer = SequentialMeanPoolingLayer()
                lr_layer = LogisticRegressionLayer(lr_W_init, lambda: lr_b_init()[0])
                probs = th_x
                for layer in [lstm_layer, smp_layer, lr_layer]:
                    probs = layer.get_output_expr(probs)
                loss = T.mean(T.nnet.binary_crossentropy(probs, th_true_labels))
                grads = T.grad(loss, wrt=[lr_layer.W, lr_layer.b, lstm_layer.W, lstm_layer.R, th_x])
                get_theano_grads = theano.function([th_x, th_true_labels], grads)
                th_x = np.dstack(x)
                theano_grads = get_theano_grads(th_x[..., :sequence_len], true_labels)

                # quagga model
                self.rng.set_state(state)
                context = Context()
                x_gpu = MatrixList([Connector(Matrix.from_npa(e), context, context) for e in x])
                true_labels_gpu = Connector(Matrix.from_npa(true_labels))
                lstm_block = LstmRnn(W_init, R_init, x_gpu, reverse)
                smp_block = SequentialMeanPoolingBlock(lstm_block.h)
                dot_block = DotBlock(lr_W_init, lr_b_init, smp_block.output)
                sce_block = SigmoidCeBlock(dot_block.output, true_labels_gpu)
                x_gpu.set_length(sequence_len)
                for e in x_gpu:
                    e.fprop()
                true_labels_gpu.fprop()
                lstm_block.fprop()
                smp_block.fprop()
                dot_block.fprop()
                sce_block.fprop()
                sce_block.bprop()
                dot_block.bprop()
                smp_block.bprop()
                lstm_block.bprop()
                cudart.cuda_device_synchronize()
                quagga_grads = [dot_block.dL_dW.to_host(),
                                dot_block.dL_db.to_host(),
                                lstm_block.dL_dW.to_host(),
                                lstm_block.dL_dR.to_host(),
                                [e.backward_matrix.to_host() for e in x_gpu]]

                for theano_grad, quagga_grad in izip(theano_grads[:-1], quagga_grads[:-1]):
                    r.append(np.allclose(theano_grad, quagga_grad))
                for i in xrange(theano_grads[-1].shape[-1]):
                    if not np.allclose(quagga_grads[-1][i], theano_grads[-1][..., i]):
                        r.append(False)
                        break
                else:
                    r.append(True)

                del lstm_block
                del dot_block
                del smp_block
                del sce_block

        self.assertEqual(sum(r), self.N * 10)