import quagga
import theano
import numpy as np
from itertools import izip
from unittest import TestCase
from theano import tensor as T
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import DotBlock
from quagga.connector import Connector
from quagga.matrix import MatrixList
from quagga.blocks import SelectorBlock
from quagga.blocks import SigmoidCeBlock


class TestSelectorBlock(TestCase):
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
            batch_size = self.rng.random_integers(512)
            dim = self.rng.random_integers(1500)
            x = [self.rng.rand(batch_size, dim).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)

            for j in xrange(self.N):
                state = self.rng.get_state()
                quagga.processor_type = 'gpu'
                x_gpu = MatrixList([Connector(Matrix.from_npa(e)) for e in x])
                selector_block_gpu = SelectorBlock(x_gpu)
                x_gpu.set_length(sequence_len)
                selector_block_gpu.fprop(self.rng.randint(sequence_len))
                output_gpu = selector_block_gpu.output.to_host()

                self.rng.set_state(state)
                quagga.processor_type = 'cpu'
                x_cpu = MatrixList([Connector(Matrix.from_npa(e)) for e in x])
                selector_block_cpu = SelectorBlock(x_cpu)
                x_cpu.set_length(sequence_len)
                selector_block_cpu.fprop(self.rng.randint(sequence_len))
                output_cpu = selector_block_cpu.output.to_host()

                r.append(np.allclose(output_gpu, output_cpu))

        self.assertEqual(sum(r), self.N * self.N)

    def test_bprop(self):
        """
        compare `bprop` results for cpu and gpu backends
        """
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(500)
            batch_size = self.rng.random_integers(512)
            dim = self.rng.random_integers(1500)
            x = [self.rng.rand(batch_size, dim).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)

            for j in xrange(self.N):
                state = self.rng.get_state()
                quagga.processor_type = 'gpu'
                context = Context()
                x_gpu = MatrixList([Connector(Matrix.from_npa(e), context, context) for e in x])
                selector_block_gpu = SelectorBlock(x_gpu)
                x_gpu.set_length(sequence_len)
                _, dL_doutput = selector_block_gpu.output.register_usage(context, context)
                selector_block_gpu.fprop(self.rng.randint(sequence_len))
                random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
                Matrix.from_npa(random_matrix, 'float').copy_to(context, dL_doutput)
                selector_block_gpu.bprop()
                dL_dx_gpu = [e.backward_matrix.to_host() for e in x_gpu]

                self.rng.set_state(state)
                quagga.processor_type = 'cpu'
                context = Context()
                x_cpu = MatrixList([Connector(Matrix.from_npa(e), context, context) for e in x])
                selector_block_cpu = SelectorBlock(x_cpu)
                x_cpu.set_length(sequence_len)
                _, dL_doutput = selector_block_cpu.output.register_usage(context, context)
                selector_block_cpu.fprop(self.rng.randint(sequence_len))
                random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
                Matrix.from_npa(random_matrix, 'float').copy_to(context, dL_doutput)
                selector_block_cpu.bprop()
                dL_dx_cpu = [e.backward_matrix.to_host() for e in x_cpu]

                for dL_dx_gpu, dL_dx_cpu in izip(dL_dx_gpu, dL_dx_cpu):
                    if not np.allclose(dL_dx_gpu, dL_dx_cpu):
                        r.append(False)
                        break
                else:
                    r.append(True)

        self.assertEqual(sum(r), self.N * self.N)

    def test_theano_grad(self):
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
            batch_size = self.rng.random_integers(512)
            dim = self.rng.random_integers(1500)
            x = [self.rng.rand(batch_size, dim).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            true_labels = self.rng.randint(1, size=(batch_size, 1)).astype(dtype=np.float32)

            W_init = self.get_orthogonal_initializer(dim, 1)
            b_init = lambda: self.rng.rand(1, 1).astype(dtype=np.float32)

            # Theano model
            state = self.rng.get_state()
            th_index = T.iscalar()
            th_x = T.ftensor3()
            th_x_value = np.dstack([e for e in x[:sequence_len]])
            th_true_labels = T.fmatrix()
            lr_layer = LogisticRegressionLayer(W_init, lambda: b_init()[0])
            probs = lr_layer.get_output_expr(th_x[:, :, th_index])
            loss = T.mean(T.nnet.binary_crossentropy(probs, th_true_labels))
            grad_x = T.grad(loss, wrt=th_x)
            get_grad_x = theano.function([th_index, th_x, th_true_labels], grad_x)

            # quagga model
            self.rng.set_state(state)
            context = Context()
            x_gpu = MatrixList([Connector(Matrix.from_npa(e), context, context) for e in x])
            x_gpu.set_length(sequence_len)
            true_labels_gpu = Connector(Matrix.from_npa(true_labels))
            selector_block = SelectorBlock(x_gpu)
            dot_block = DotBlock(W_init, b_init, selector_block.output)
            sce_block = SigmoidCeBlock(dot_block.output, true_labels_gpu)

            for j in xrange(self.N):
                index = self.rng.randint(sequence_len)
                selector_block.fprop(index)
                dot_block.fprop()
                sce_block.fprop()
                sce_block.bprop()
                dot_block.bprop()
                selector_block.bprop()

                dL_dx_gpu = [e.backward_matrix.to_host() for e in x_gpu]
                dL_dx_th = get_grad_x(index, th_x_value, true_labels)

                for i in xrange(dL_dx_th.shape[-1]):
                    if not np.allclose(dL_dx_gpu[i], dL_dx_th[..., i]):
                        r.append(False)
                        break
                else:
                    r.append(True)

        self.assertEqual(sum(r), self.N * self.N)