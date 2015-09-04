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
from quagga.blocks import SigmoidCeBlock
from quagga.blocks import SequentialEmbeddingBlock
from quagga.blocks import SequentialMeanPoolingBlock


class TestSequentialEmbeddingBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 10

    @classmethod
    def get_random_array(cls, shape=None):
        if shape:
            a = 4 * cls.rng.rand(*shape) - 2
        else:
            nrows, ncols = cls.rng.randint(low=1, high=7000, size=2)
            a = 4 * cls.rng.rand(nrows, ncols) - 2
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
        r = []
        for i in xrange(self.N):
            batch_size = self.rng.random_integers(512)
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            vocab_size = self.rng.random_integers(50000)
            dim = self.rng.random_integers(1500)
            x = self.rng.randint(vocab_size, size=(batch_size, max_input_sequence_len)).astype(dtype=np.int32)
            embedding_init = lambda: self.rng.randn(vocab_size, dim).astype(dtype=np.float32)

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            x_gpu = Connector(Matrix.from_npa(x))
            seq_embd_block = SequentialEmbeddingBlock(embedding_init, x_gpu, learning=False)
            x_gpu.ncols = sequence_len
            x_gpu.fprop()
            seq_embd_block.fprop()
            output_gpu = seq_embd_block.output.to_host()

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            x_cpu = Connector(Matrix.from_npa(x))
            seq_embd_block = SequentialEmbeddingBlock(embedding_init, x_cpu, learning=False)
            x_cpu.ncols = sequence_len
            x_cpu.fprop()
            seq_embd_block.fprop()
            output_cpu = seq_embd_block.output.to_host()

            for output_gpu, output_cpu in izip(output_gpu, output_cpu):
                if not np.allclose(output_gpu, output_cpu):
                    r.append(False)
                    break
            else:
                r.append(True)

        self.assertEqual(sum(r), self.N)

    def test_bprop(self):
        r = []
        for i in xrange(self.N):
            batch_size = self.rng.random_integers(512)
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            vocab_size = self.rng.random_integers(50000)
            dim = self.rng.random_integers(1500)
            x = self.rng.randint(vocab_size, size=(batch_size, max_input_sequence_len)).astype(dtype=np.int32)
            embedding_init = lambda: self.rng.randn(vocab_size, dim).astype(dtype=np.float32)

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            context = Context()
            x_gpu = Connector(Matrix.from_npa(x), context)
            seq_embd_block = SequentialEmbeddingBlock(embedding_init, x_gpu)
            _, dL_doutput = zip(*[e.register_usage(context, context) for e in seq_embd_block.output])
            x_gpu.ncols = sequence_len
            x_gpu.fprop()
            seq_embd_block.fprop()
            for _, dL_doutput in izip(_, dL_doutput):
                random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
                Matrix.from_npa(random_matrix, 'float').copy_to(context, dL_doutput)
            seq_embd_block.bprop()
            dL_embedding_gpu = [e.to_host() for e in seq_embd_block.dL_embedding]

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            context = Context()
            x_cpu = Connector(Matrix.from_npa(x), context)
            seq_embd_block = SequentialEmbeddingBlock(embedding_init, x_cpu)
            _, dL_doutput = zip(*[e.register_usage(context, context) for e in seq_embd_block.output])
            x_cpu.ncols = sequence_len
            x_cpu.fprop()
            seq_embd_block.fprop()
            for _, dL_doutput in izip(_, dL_doutput):
                random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
                Matrix.from_npa(random_matrix, 'float').copy_to(context, dL_doutput)
            seq_embd_block.bprop()
            dL_embedding_cpu = [e.to_host() for e in seq_embd_block.dL_embedding]

            for dL_embedding_gpu, dL_embedding_cpu in izip(dL_embedding_gpu, dL_embedding_cpu):
                if not np.allclose(dL_embedding_gpu, dL_embedding_cpu):
                    r.append(False)
                    break
            else:
                r.append(True)

        self.assertEqual(sum(r), self.N)

    def test_theano_grad(self):
        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            batch_size = self.rng.random_integers(512)
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            vocab_size = self.rng.random_integers(50000)
            dim = self.rng.random_integers(1500)
            x = self.rng.randint(vocab_size, size=(batch_size, max_input_sequence_len)).astype(dtype=np.int32)
            embedding_init = lambda: self.rng.randn(vocab_size, dim).astype(dtype=np.float32)
            true_labels = self.rng.randint(2, size=(batch_size, 1)).astype(dtype=np.float32)
            lr_W_init = self.get_orthogonal_initializer(dim, 1)
            lr_b_init = lambda: self.rng.rand(1, 1).astype(dtype=np.float32)

            # Theano model
            state = self.rng.get_state()
            th_x = T.imatrix()
            th_true_labels = T.fmatrix()
            embed_layer = EmbeddingLayer(embedding_init)
            lr_layer = LogisticRegressionLayer(lr_W_init, lambda: lr_b_init()[0])
            probs = embed_layer.get_output_expr(th_x)
            probs = T.mean(probs, axis=1)
            probs = lr_layer.get_output_expr(probs)
            loss = T.mean(T.nnet.binary_crossentropy(probs, th_true_labels))
            grads = T.grad(loss, wrt=[lr_layer.W, lr_layer.b, embed_layer.W])
            updates = [(embed_layer.W, embed_layer.W + grads[-1])]
            get_theano_grads = theano.function([th_x, th_true_labels], grads[:-1], updates=updates)
            theano_grads = get_theano_grads(x[:, :sequence_len], true_labels)

            # quagga model
            self.rng.set_state(state)
            x_gpu = Connector(Matrix.from_npa(x))
            x_gpu.ncols = sequence_len
            true_labels_gpu = Connector(Matrix.from_npa(true_labels))
            sembd_block = SequentialEmbeddingBlock(embedding_init, x_gpu)
            smp_block = SequentialMeanPoolingBlock(sembd_block.output)
            dot_block = DotBlock(lr_W_init, lr_b_init, smp_block.output)
            sce_block = SigmoidCeBlock(dot_block.output, true_labels_gpu)
            sembd_block.fprop()
            smp_block.fprop()
            dot_block.fprop()
            sce_block.fprop()
            sce_block.bprop()
            dot_block.bprop()
            smp_block.bprop()
            sembd_block.bprop()
            sembd_block.embedding.sliced_rows_batch_scaled_add(sembd_block.context, x_gpu, 1.0, sembd_block.dL_embedding)
            quagga_grads = [dot_block.dL_dW.to_host(), dot_block.dL_db.to_host()]

            for theano_grad, quagga_grad in izip(theano_grads, quagga_grads):
                r.append(np.allclose(theano_grad, quagga_grad))
            r.append(np.allclose(embed_layer.W.get_value(), sembd_block.embedding.to_host()))

            del sembd_block
            del smp_block
            del dot_block
            del sce_block

        self.assertEqual(sum(r), self.N * 3)


class LogisticRegressionLayer(object):
    def __init__(self, W_init, b_init):
        self.W = theano.shared(value=W_init())
        self.b = theano.shared(value=b_init())

    def get_output_expr(self, input_expr):
        return T.nnet.sigmoid(T.dot(input_expr, self.W) + self.b)


class EmbeddingLayer(object):
    def __init__(self, W_init):
        self.W = theano.shared(value=W_init())

    def get_output_expr(self, input_expr):
        return self.W[input_expr]