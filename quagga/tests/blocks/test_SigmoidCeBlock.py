import quagga
import theano
import numpy as np
from unittest import TestCase
from theano import tensor as T
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector
from quagga.blocks import SigmoidCeBlock


class TestSigmoidCeBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 10

    def test_fprop(self):
        """
        compare `fprop` results for cpu and gpu backends
        """
        r = []
        for i in xrange(self.N):
            batch_size = self.rng.random_integers(2000)
            true_labels = self.rng.randint(2, size=(batch_size, 1)).astype(dtype=np.float32)
            x = self.rng.randn(batch_size, 1).astype(dtype=np.float32)

            quagga.processor_type = 'gpu'
            x_gpu = Connector(Matrix.from_npa(x))
            true_labels_gpu = Connector(Matrix.from_npa(true_labels))
            sigmoid_ce_block = SigmoidCeBlock(x_gpu, true_labels_gpu)
            sigmoid_ce_block.fprop()
            probs_gpu = sigmoid_ce_block.probs.to_host()

            quagga.processor_type = 'cpu'
            x_cpu = Connector(Matrix.from_npa(x))
            true_labels_cpu = Connector(Matrix.from_npa(true_labels))
            sigmoid_ce_block = SigmoidCeBlock(x_cpu, true_labels_cpu)
            sigmoid_ce_block.fprop()
            probs_cpu = sigmoid_ce_block.probs.to_host()

            r.append(np.allclose(probs_gpu, probs_cpu))

        self.assertEqual(sum(r), self.N)

    def test_bprop(self):
        """
        compare `bprop` results for cpu and gpu backends
        """
        r = []
        for i in xrange(self.N):
            batch_size = self.rng.random_integers(2000)
            true_labels = self.rng.randint(2, size=(batch_size, 1)).astype(dtype=np.float32)
            x = self.rng.randn(batch_size, 1).astype(dtype=np.float32)

            quagga.processor_type = 'gpu'
            context = Context()
            x_gpu = Connector(Matrix.from_npa(x), context, context)
            true_labels_gpu = Connector(Matrix.from_npa(true_labels))
            sigmoid_ce_block = SigmoidCeBlock(x_gpu, true_labels_gpu)
            sigmoid_ce_block.fprop()
            sigmoid_ce_block.bprop()
            dL_dx_gpu = x_gpu.backward_matrix.to_host()

            context = Context()
            x_cpu = Connector(Matrix.from_npa(x), context, context)
            true_labels_cpu = Connector(Matrix.from_npa(true_labels))
            sigmoid_ce_block = SigmoidCeBlock(x_cpu, true_labels_cpu)
            sigmoid_ce_block.fprop()
            sigmoid_ce_block.bprop()
            dL_dx_cpu = x_cpu.backward_matrix.to_host()

            r.append(np.allclose(dL_dx_gpu, dL_dx_cpu))

        self.assertEqual(sum(r), self.N)

    def test_theano_grad(self):
        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            batch_size = self.rng.random_integers(2000)
            true_labels = self.rng.randint(2, size=(batch_size, 1)).astype(dtype=np.float32)
            x = self.rng.randn(batch_size, 1).astype(dtype=np.float32)

            # Theano model
            th_x = T.fmatrix()
            th_true_labels = T.fmatrix()
            probs = T.nnet.sigmoid(th_x)
            loss = T.mean(T.nnet.binary_crossentropy(probs, th_true_labels))
            get_theano_grads = theano.function([th_x, th_true_labels], T.grad(loss, wrt=th_x))
            th_dL_dx = get_theano_grads(x, true_labels)

            # quagga model
            context = Context()
            x_gpu = Connector(Matrix.from_npa(x), context, context)
            true_labels_gpu = Connector(Matrix.from_npa(true_labels))
            sigmoid_ce_block = SigmoidCeBlock(x_gpu, true_labels_gpu)
            sigmoid_ce_block.fprop()
            sigmoid_ce_block.bprop()
            q_dL_dx = x_gpu.backward_matrix.to_host()

            r.append(np.allclose(th_dL_dx, q_dL_dx))

        self.assertEqual(sum(r), self.N)