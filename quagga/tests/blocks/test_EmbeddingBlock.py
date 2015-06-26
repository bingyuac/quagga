import numpy as np
from unittest import TestCase
from quagga.context import Context
from quagga.matrix import GpuMatrix
from quagga.connector import Connector
from quagga.blocks import EmbeddingBlock


class testEmbeddingBlock(TestCase):
    def test_fprop(self):
        r = []
        n = 100

        for i in xrange(n):
            m = 10
            k = 20
            dim = 100

            indexes_np = np.random.randint(m, size=(1, k))
            indexes = Connector(GpuMatrix.from_npa(indexes_np, 'int'))
            embedding_np = (np.random.rand(dim, m) * 0.1).astype(np.float32)
            embedding_init = lambda: embedding_np
            embed_block = EmbeddingBlock(embedding_init, indexes)
            embed_block.fprop()
            a = embed_block.output.to_host()
            b = embedding_np[:, indexes_np.flatten()]
            r.append(np.allclose(a, b))
        self.assertEqual(sum(r), n)

    def test_bprop(self):
        r = []
        n = 100

        context = Context()
        for i in xrange(n):
            m = 10
            k = 20
            dim = 100

            indexes_np = np.random.randint(m, size=(1, k))
            indexes = Connector(GpuMatrix.from_npa(indexes_np, 'int'))
            embedding_np = (np.random.rand(dim, m) * 0.1).astype(np.float32)
            embedding_init = lambda: embedding_np
            embed_block = EmbeddingBlock(embedding_init, indexes)
            output, dL_doutput = embed_block.output.register_usage(context, context)
            embed_block.fprop()
            GpuMatrix.from_npa(np.random.rand(dL_doutput.nrows, dL_doutput.ncols), 'float').copy(context, dL_doutput)
            embed_block.bprop()
            sparse_grad = embed_block.grads[0][1]
            a, b = sparse_grad[0].to_host(), sparse_grad[1].to_host()
            r.append(np.allclose(a, indexes_np))
            r.append(np.allclose(b, dL_doutput.to_host()))

        self.assertEqual(sum(r), 2 * n)
