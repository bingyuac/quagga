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
from quagga.blocks import SigmoidCeBlock
from quagga.blocks import SequentialMeanPoolingBlock


class TestSequentialMeanPoolingBlock(TestCase):
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
            dim = self.rng.random_integers(1500)
            x = [self.rng.rand(batch_size, dim).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            x_gpu = MatrixList([Connector(Matrix.from_npa(e)) for e in x])
            smean_pooling_block_gpu = SequentialMeanPoolingBlock(x_gpu)
            x_gpu.set_length(sequence_len)
            smean_pooling_block_gpu.fprop()
            output_gpu = smean_pooling_block_gpu.output.to_host()

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            x_cpu = MatrixList([Connector(Matrix.from_npa(e)) for e in x])
            smean_pooling_block_cpu = SequentialMeanPoolingBlock(x_cpu)
            x_cpu.set_length(sequence_len)
            smean_pooling_block_cpu.fprop()
            output_cpu = smean_pooling_block_cpu.output.to_host()

            r.append(np.allclose(output_gpu, output_cpu))

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
            dim = self.rng.random_integers(1500)
            x = [self.rng.rand(batch_size, dim).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            context = Context()
            x_gpu = MatrixList([Connector(Matrix.from_npa(e), context, context) for e in x])
            smean_pooling_block_gpu = SequentialMeanPoolingBlock(x_gpu)
            x_gpu.set_length(sequence_len)
            _, dL_doutput = smean_pooling_block_gpu.output.register_usage(context, context)
            smean_pooling_block_gpu.fprop()
            random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
            Matrix.from_npa(random_matrix, 'float').copy_to(context, dL_doutput)
            smean_pooling_block_gpu.bprop()
            dL_dmatrices_gpu = [e.backward_matrix.to_host() for e in x_gpu]

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            context = Context()
            x_cpu = MatrixList([Connector(Matrix.from_npa(e), context, context) for e in x])
            smean_pooling_block_cpu = SequentialMeanPoolingBlock(x_cpu)
            x_cpu.set_length(sequence_len)
            _, dL_doutput = smean_pooling_block_cpu.output.register_usage(context, context)
            smean_pooling_block_cpu.fprop()
            random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
            Matrix.from_npa(random_matrix, 'float').copy_to(context, dL_doutput)
            smean_pooling_block_cpu.bprop()
            dL_dmatrices_cpu = [e.backward_matrix.to_host() for e in x_cpu]

            for dL_dmatrix_gpu, dL_dmatrix_cpu in izip(dL_dmatrices_gpu, dL_dmatrices_cpu):
                if not np.allclose(dL_dmatrix_gpu, dL_dmatrix_cpu):
                    r.append(False)
                    break
            else:
                r.append(True)

        self.assertEqual(sum(r), self.N)

    def test_theano_grad(self):
        class SequentialMeanPoolingLayer(object):
            def get_output_expr(self, input_sequence):
                return T.mean(input_sequence, axis=2)

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
            batch_size = self.rng.random_integers(512)
            dim = self.rng.random_integers(1500)
            x = [self.rng.rand(batch_size, dim).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]
            true_labels = self.rng.randint(1, size=(batch_size, 1)).astype(dtype=np.float32)

            W_init = self.get_orthogonal_initializer(dim, 1)
            b_init = lambda: self.rng.rand(1, 1).astype(dtype=np.float32)

            # Theano model
            state = self.rng.get_state()
            th_x = T.ftensor3()
            th_true_labels = T.fmatrix()
            smp_layer = SequentialMeanPoolingLayer()
            lr_layer = LogisticRegressionLayer(W_init, lambda: b_init()[0])
            probs = lr_layer.get_output_expr(smp_layer.get_output_expr(th_x))
            loss = T.mean(T.nnet.binary_crossentropy(probs, th_true_labels))
            grad_x = T.grad(loss, wrt=th_x)
            get_grad_x = theano.function([th_x, th_true_labels], grad_x)

            # quagga model
            self.rng.set_state(state)
            context = Context()
            x = MatrixList([Connector(Matrix.from_npa(e), context, context) for e in x])
            true_labels = Connector(Matrix.from_npa(true_labels))
            smp_block = SequentialMeanPoolingBlock(x)
            dot_block = DotBlock(W_init, b_init, smp_block.output)
            sce_block = SigmoidCeBlock(dot_block.output, true_labels)
            x.set_length(sequence_len)
            smp_block.fprop()
            dot_block.fprop()
            sce_block.fprop()
            sce_block.bprop()
            dot_block.bprop()
            smp_block.bprop()

            dL_dx = [e.backward_matrix.to_host() for e in x]
            dL_dx_th = get_grad_x(np.dstack([e.to_host() for e in x]), true_labels.to_host())
            for i in xrange(dL_dx_th.shape[-1]):
                if not np.allclose(dL_dx[i], dL_dx_th[..., i]):
                    r.append(False)
                    break
            else:
                r.append(True)

        self.assertEqual(sum(r), self.N)
