import quagga
import theano
import numpy as np
from unittest import TestCase
from theano import tensor as T
from quagga.matrix import Matrix
from quagga.connector import Connector
from quagga.blocks import SoftmaxCeBlock


class TestSoftmaxCeBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 50

    def test_fprop(self):
        """
        compare `fprop` results for cpu and gpu backends
        """
        r = []
        for i in xrange(self.N):
            for sparse in [False, True]:
                batch_size, dim = self.rng.random_integers(2000, size=2)
                if sparse:
                    true_labels = np.zeros((batch_size, dim), np.float32)
                    for k, j in enumerate(self.rng.randint(dim, size=batch_size)):
                        true_labels[k, j] = 1.0
                else:
                    true_labels = self.rng.randint(dim, size=(batch_size, 1)).astype(np.int32)
                x = self.rng.randn(batch_size, dim).astype(np.float32)
                mask = (self.rng.rand(batch_size, 1) < 0.8).astype(np.float32)
                for with_mask in [False, True]:
                    quagga.processor_type = 'gpu'
                    x_gpu = Connector(Matrix.from_npa(x))
                    true_labels_gpu = Connector(Matrix.from_npa(true_labels))
                    mask_gpu = Connector(Matrix.from_npa(mask)) if with_mask else None
                    softmax_ce_block = SoftmaxCeBlock(x_gpu, true_labels_gpu, mask_gpu)
                    x_gpu.fprop()
                    true_labels_gpu.fprop()
                    if with_mask:
                        mask_gpu.fprop()
                    softmax_ce_block.fprop()
                    probs_gpu = softmax_ce_block.probs.to_host()

                    quagga.processor_type = 'cpu'
                    x_cpu = Connector(Matrix.from_npa(x))
                    true_labels_cpu = Connector(Matrix.from_npa(true_labels))
                    mask_cpu = Connector(Matrix.from_npa(mask)) if with_mask else None
                    softmax_ce_block = SoftmaxCeBlock(x_cpu, true_labels_cpu, mask_cpu)
                    x_cpu.fprop()
                    true_labels_cpu.fprop()
                    if with_mask:
                        mask_cpu.fprop()
                    softmax_ce_block.fprop()
                    probs_cpu = softmax_ce_block.probs.to_host()

                    r.append(np.allclose(probs_gpu, probs_cpu))

        self.assertEqual(sum(r), len(r))

    def test_bprop(self):
        """
        compare `bprop` results for cpu and gpu backends
        """
        r = []
        for i in xrange(self.N):
            for sparse in [True, False]:
                batch_size, dim = self.rng.random_integers(2000, size=2)
                if sparse:
                    true_labels = np.zeros((batch_size, dim), np.float32)
                    for k, j in enumerate(self.rng.randint(dim, size=batch_size)):
                        true_labels[k, j] = 1.0
                else:
                    true_labels = self.rng.randint(dim, size=(batch_size, 1)).astype(np.int32)
                x = self.rng.randn(batch_size, dim).astype(np.float32)
                mask = (self.rng.rand(batch_size, 1) < 0.8).astype(np.float32)
                device_id = 0
                for with_mask in [False, True]:
                    quagga.processor_type = 'gpu'
                    x_gpu = Connector(Matrix.from_npa(x), device_id)
                    true_labels_gpu = Connector(Matrix.from_npa(true_labels))
                    mask_gpu = Connector(Matrix.from_npa(mask)) if with_mask else None
                    softmax_ce_block = SoftmaxCeBlock(x_gpu, true_labels_gpu, mask_gpu)
                    x_gpu.fprop()
                    true_labels_gpu.fprop()
                    if with_mask:
                        mask_gpu.fprop()
                    softmax_ce_block.fprop()
                    softmax_ce_block.bprop()
                    dL_dx_gpu = x_gpu.backward_matrix.to_host()

                    quagga.processor_type = 'cpu'
                    x_cpu = Connector(Matrix.from_npa(x), device_id)
                    true_labels_cpu = Connector(Matrix.from_npa(true_labels))
                    mask_cpu = Connector(Matrix.from_npa(mask)) if with_mask else None
                    softmax_ce_block = SoftmaxCeBlock(x_cpu, true_labels_cpu, mask_cpu)
                    x_cpu.fprop()
                    true_labels_cpu.fprop()
                    if with_mask:
                        mask_cpu.fprop()
                    softmax_ce_block.fprop()
                    softmax_ce_block.bprop()
                    dL_dx_cpu = x_cpu.backward_matrix.to_host()

                    r.append(np.allclose(dL_dx_gpu, dL_dx_cpu))

        self.assertEqual(sum(r), len(r))

    def test_theano_grad(self):
        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            for sparse in [True, False]:
                batch_size, dim = self.rng.random_integers(2000, size=2)
                if sparse:
                    true_labels = np.zeros((batch_size, dim), np.float32)
                    for k, j in enumerate(self.rng.randint(dim, size=batch_size)):
                        true_labels[k, j] = 1.0
                else:
                    true_labels = self.rng.randint(dim, size=(batch_size, 1)).astype(np.int32)
                x = self.rng.randn(batch_size, dim).astype(np.float32)
                mask = (self.rng.rand(batch_size, 1) < 0.8).astype(np.float32)
                device_id = 0
                for with_mask in [False, True]:
                    # Theano model
                    th_x = T.fmatrix()
                    th_mask = T.fcol()
                    th_true_labels = T.fmatrix() if sparse else T.ivector()
                    if with_mask:
                        probs = T.nnet.softmax(th_mask * th_x)
                    else:
                        probs = T.nnet.softmax(th_x)
                    loss = T.mean(T.nnet.categorical_crossentropy(probs, th_true_labels))
                    if with_mask:
                        get_theano_grads = theano.function([th_x, th_true_labels, th_mask], T.grad(loss, wrt=th_x))
                        th_dL_dx = get_theano_grads(x, true_labels if sparse else true_labels[:, 0], mask)
                    else:
                        get_theano_grads = theano.function([th_x, th_true_labels], T.grad(loss, wrt=th_x))
                        th_dL_dx = get_theano_grads(x, true_labels if sparse else true_labels[:, 0])

                    # quagga model
                    x_gpu = Connector(Matrix.from_npa(x), device_id)
                    true_labels_gpu = Connector(Matrix.from_npa(true_labels))
                    mask_gpu = Connector(Matrix.from_npa(mask)) if with_mask else None
                    softmax_ce_block = SoftmaxCeBlock(x_gpu, true_labels_gpu, mask_gpu)
                    x_gpu.fprop()
                    true_labels_gpu.fprop()
                    if with_mask:
                        mask_gpu.fprop()
                    softmax_ce_block.fprop()
                    softmax_ce_block.bprop()
                    q_dL_dx = x_gpu.backward_matrix.to_host()

                    r.append(np.allclose(th_dL_dx, q_dL_dx))

        self.assertEqual(sum(r), len(r))