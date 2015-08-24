import quagga
import numpy as np
from itertools import izip
from unittest import TestCase
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector
from quagga.blocks import SequentialEmbeddingBlock


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

            for reverse in [False, True]:
                state = self.rng.get_state()
                quagga.processor_type = 'gpu'
                x_gpu = Connector(Matrix.from_npa(x))
                seq_embd_block = SequentialEmbeddingBlock(embedding_init, x_gpu, learning=False, reverse=reverse)
                x_gpu.ncols = sequence_len
                x_gpu.fprop()
                seq_embd_block.fprop()
                seq_embd_block.context.synchronize()
                output_gpu = seq_embd_block.output

                self.rng.set_state(state)
                quagga.processor_type = 'cpu'
                x_cpu = Connector(Matrix.from_npa(x))
                seq_embd_block = SequentialEmbeddingBlock(embedding_init, x_cpu, learning=False, reverse=reverse)
                x_cpu.ncols = sequence_len
                x_cpu.fprop()
                seq_embd_block.fprop()
                seq_embd_block.context.synchronize()
                output_cpu = seq_embd_block.output

                for output_gpu, output_cpu in izip(output_gpu, output_cpu):
                    if not np.allclose(output_gpu.to_host(), output_cpu.to_host()):
                        r.append(False)
                        break
                else:
                    r.append(True)

        self.assertEqual(sum(r), self.N * 2)

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

            for reverse in [False, True]:
                state = self.rng.get_state()
                quagga.processor_type = 'gpu'
                context = Context()
                x_gpu = Connector(Matrix.from_npa(x), context)
                seq_embd_block = SequentialEmbeddingBlock(embedding_init, x_gpu, reverse=reverse)
                _, dL_doutput = zip(*[e.register_usage(context, context) for e in seq_embd_block.output])
                x_gpu.ncols = sequence_len
                x_gpu.fprop()
                seq_embd_block.fprop()
                for _, dL_doutput in izip(_, dL_doutput):
                    random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
                    Matrix.from_npa(random_matrix, 'float').copy(context, dL_doutput)
                seq_embd_block.bprop()
                [(context, (indexes_gpu, dL_embedding_gpu))] = seq_embd_block.grads
                context.synchronize()

                self.rng.set_state(state)
                quagga.processor_type = 'cpu'
                context = Context()
                x_cpu = Connector(Matrix.from_npa(x), context)
                seq_embd_block = SequentialEmbeddingBlock(embedding_init, x_cpu, reverse=reverse)
                _, dL_doutput = zip(*[e.register_usage(context, context) for e in seq_embd_block.output])
                x_cpu.ncols = sequence_len
                x_cpu.fprop()
                seq_embd_block.fprop()
                for _, dL_doutput in izip(_, dL_doutput):
                    random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
                    Matrix.from_npa(random_matrix, 'float').copy(context, dL_doutput)
                seq_embd_block.bprop()
                [(context, (indexes_cpu, dL_embedding_cpu))] = seq_embd_block.grads
                context.synchronize()

                r.append(np.allclose(indexes_gpu.to_host(), indexes_cpu.to_host()))
                r.append(np.allclose(dL_embedding_gpu.to_host(), dL_embedding_cpu.to_host()))

        self.assertEqual(sum(r), self.N * 4)