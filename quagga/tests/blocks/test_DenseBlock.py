import numpy as np
from unittest import TestCase
from quagga.context import Context
from quagga.matrix import GpuMatrix
from quagga.connector import Connector
from quagga.blocks import DenseBlock, LogisticRegressionCe


class TestDenseBlock(TestCase):
    def test_finite_difference_parameters(self):
        r = []
        n = 10

        for activation_fun in ['sigmoid', 'tanh', 'relu']:
            for i in xrange(n):
                k = 3
                dim = 50

                dense_init = lambda: (np.random.rand(dim, dim) * 0.1).astype(np.float32)
                log_reg_init = lambda: (np.random.rand(1, dim) * 0.1).astype(np.float32)
                features = Connector(GpuMatrix.from_npa(np.random.rand(dim, k), 'float'))
                true_labels = Connector(GpuMatrix.from_npa(np.array([[0, 1, 0]], np.float32)))
                dense_block = DenseBlock(dense_init, features, activation_fun)
                dense_block.set_training_mode()
                log_reg = LogisticRegressionCe(log_reg_init, dense_block.output, true_labels)
                w_np = dense_block.w.to_host()
                true_labels = true_labels.to_host()

                dense_block.fprop()
                log_reg.fprop()
                log_reg.bprop()
                dense_block.bprop()
                dL_dw = dense_block.dL_dw.to_host()
                numerical_grad = np.zeros_like(dL_dw)
                cross_entropy = lambda l, p: -np.sum(l * np.log(p) + (1 - l) * np.log(1 - p))

                epsilon = 1E-2
                for i in xrange(dense_block.w.nrows):
                    for j in xrange(dense_block.w.ncols):
                        dense_block.w[i, j] = w_np[i, j] + epsilon
                        dense_block.fprop()
                        log_reg.fprop()
                        probs = log_reg.probs.to_host()
                        plus_cost = cross_entropy(true_labels, probs)

                        dense_block.w[i, j] = w_np[i, j] - epsilon
                        dense_block.fprop()
                        log_reg.fprop()
                        probs = log_reg.probs.to_host()
                        minus_cost = cross_entropy(true_labels, probs)

                        numerical_grad[i, j] = (plus_cost - minus_cost) / (2 * epsilon)
                        dense_block.w[i, j] = w_np[i, j]

                r.append(np.allclose(dL_dw, numerical_grad, atol=1e-3))

        self.assertEqual(sum(r), 3 * n)

    def test_finite_difference_features(self):
        r = []
        n = 10

        for activation_fun in ['sigmoid', 'tanh', 'relu']:
            for i in xrange(n):
                k = 3
                dim = 50

                dense_init = lambda: (np.random.rand(dim, dim) * 0.1).astype(np.float32)
                log_reg_init = lambda: (np.random.rand(1, dim) * 0.1).astype(np.float32)
                features = Connector(GpuMatrix.from_npa(np.random.rand(dim, k), 'float'), b_usage_context=Context())
                true_labels = Connector(GpuMatrix.from_npa(np.array([[0, 1, 0]], np.float32)))
                dense_block = DenseBlock(dense_init, features, activation_fun)
                dense_block.set_training_mode()
                log_reg = LogisticRegressionCe(log_reg_init, dense_block.output, true_labels)

                features.fprop()
                true_labels.fprop()
                dense_block.fprop()
                log_reg.fprop()
                log_reg.bprop()
                dense_block.bprop()
                dL_dfeatures = dense_block.dL_dfeatures.to_host()
                numerical_grad = np.zeros_like(dL_dfeatures)
                cross_entropy = lambda l, p: -np.sum(l * np.log(p) + (1 - l) * np.log(1 - p))
                features_np = features.to_host()
                true_labels_np = true_labels.to_host()

                epsilon = 1e-2
                for i in xrange(features.nrows):
                    for j in xrange(features.ncols):
                        features.__setitem__((i, j), features_np[i, j] + epsilon)
                        dense_block.fprop()
                        log_reg.fprop()
                        probs = log_reg.probs.to_host()
                        plus_cost = cross_entropy(true_labels_np, probs)

                        features.__setitem__((i, j), features_np[i, j] - epsilon)
                        dense_block.fprop()
                        log_reg.fprop()
                        probs = log_reg.probs.to_host()
                        minus_cost = cross_entropy(true_labels_np, probs)

                        numerical_grad[i, j] = (plus_cost - minus_cost) / (2 * epsilon)
                        features.__setitem__((i, j), features_np[i, j])

                r.append(np.allclose(dL_dfeatures, numerical_grad, rtol=1e-7, atol=1e-3))

        self.assertEqual(sum(r), 3 * n)

    def test_finite_difference_w(self):
        r = []
        n = 10

        for activation_fun in ['sigmoid', 'tanh', 'relu']:
            for i in xrange(n):
                k = 3
                dim = 50

                dense_init = lambda: (np.random.rand(dim, dim) * 0.1).astype(np.float32)
                log_reg_init = lambda: (np.random.rand(1, dim) * 0.1).astype(np.float32)
                features = Connector(GpuMatrix.from_npa(np.random.rand(dim, k), 'float'))
                true_labels = Connector(GpuMatrix.from_npa(np.array([[0, 1, 0]], np.float32)))
                dense_block = DenseBlock(dense_init, features, activation_fun)
                dense_block.set_training_mode()
                log_reg = LogisticRegressionCe(log_reg_init, dense_block.output, true_labels)

                features.fprop()
                true_labels.fprop()
                dense_block.fprop()
                log_reg.fprop()
                log_reg.bprop()
                dense_block.bprop()
                dL_dw = dense_block.dL_dw.to_host()
                numerical_grad = np.zeros_like(dL_dw)
                cross_entropy = lambda l, p: -np.sum(l * np.log(p) + (1 - l) * np.log(1 - p))
                w_np = dense_block.w.to_host()
                true_labels_np = true_labels.to_host()

                epsilon = 1e-2
                for i in xrange(dense_block.w.nrows):
                    for j in xrange(dense_block.w.ncols):
                        dense_block.w.__setitem__((i, j), w_np[i, j] + epsilon)
                        dense_block.fprop()
                        log_reg.fprop()
                        probs = log_reg.probs.to_host()
                        plus_cost = cross_entropy(true_labels_np, probs)

                        dense_block.w.__setitem__((i, j), w_np[i, j] - epsilon)
                        dense_block.fprop()
                        log_reg.fprop()
                        probs = log_reg.probs.to_host()
                        minus_cost = cross_entropy(true_labels_np, probs)

                        numerical_grad[i, j] = (plus_cost - minus_cost) / (2 * epsilon)
                        dense_block.w.__setitem__((i, j), w_np[i, j])

                r.append(np.allclose(dL_dw, numerical_grad, rtol=1e-7, atol=1e-3))

        self.assertEqual(sum(r), 3 * n)