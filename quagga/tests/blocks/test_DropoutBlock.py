import quagga
import theano
import numpy as np
from itertools import izip
from unittest import TestCase
from quagga.cuda import cudart
from theano import tensor as T
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import DotBlock
from quagga.connector import Connector
from quagga.blocks import DropoutBlock
from quagga.blocks import SigmoidCeBlock


class TestDropoutBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 10

    def test_theano_grad(self):
        class LogisticRegressionLayer(object):
            def __init__(self, W_init, b_init):
                self.W = theano.shared(value=W_init())
                self.b = theano.shared(value=b_init()[0])

            def get_output_expr(self, input_expr):
                return T.nnet.sigmoid(T.dot(input_expr, self.W) + self.b)

        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            batch_size, x_dim = self.rng.random_integers(3000, size=2)
            x = self.rng.rand(batch_size, x_dim).astype(dtype=np.float32)
            lrdot_W_init = lambda: self.rng.rand(x_dim, 1).astype(dtype=np.float32)
            lrdot_b_init = lambda: self.rng.rand(1, 1).astype(dtype=np.float32)
            true_labels = self.rng.randint(2, size=(batch_size, 1)).astype(dtype=np.float32)
            dropout_prob = self.rng.uniform()
            seed = self.rng.randint(1000)

            # quagga model
            state = self.rng.get_state()
            context = Context()
            x_gpu = Connector(Matrix.from_npa(x), context, context)
            true_labels_gpu = Connector(Matrix.from_npa(true_labels))
            dropout_block = DropoutBlock(x_gpu, dropout_prob, seed)
            lrdot_block = DotBlock(lrdot_W_init, lrdot_b_init, dropout_block.output)
            sce_block = SigmoidCeBlock(lrdot_block.output, true_labels_gpu)
            dropout_block.fprop()
            lrdot_block.fprop()
            sce_block.fprop()
            sce_block.bprop()
            lrdot_block.bprop()
            dropout_block.bprop()
            cudart.cuda_device_synchronize()
            q_grads = [lrdot_block.dL_dW.to_host(),
                       lrdot_block.dL_db.to_host(),
                       x_gpu.backward_matrix.to_host()]
            mask = (dropout_block.output.to_host() != 0).astype(np.float32)

            # Theano model
            self.rng.set_state(state)
            th_x = T.fmatrix()
            th_true_labels = T.fmatrix()
            lr_layer = LogisticRegressionLayer(lrdot_W_init, lrdot_b_init)
            probs = lr_layer.get_output_expr(th_x * mask)
            loss = T.mean(T.nnet.binary_crossentropy(probs, th_true_labels))
            th_grads = T.grad(loss, wrt=[lr_layer.W, lr_layer.b, th_x])
            get_theano_grads = theano.function([th_x, th_true_labels], th_grads)
            th_grads = get_theano_grads(x, true_labels)

            for q_grad, th_grad in izip(q_grads, th_grads):
                r.append(np.allclose(q_grad, th_grad))

        self.assertEqual(sum(r), self.N * 3)