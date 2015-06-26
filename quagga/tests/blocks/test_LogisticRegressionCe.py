import numpy as np
from unittest import TestCase
from quagga.context import Context
from quagga.matrix import GpuMatrix
from quagga.connector import Connector
from quagga.blocks import LogisticRegressionCe


class TestLogisticRegressionCe(TestCase):
    def finite_difference_test_parameters(self):
        r = []
        n = 10

        for i in xrange(n):
            k = 3
            dim = 100

            init = lambda: (np.random.rand(1, dim) * 0.1).astype(np.float32)
            features = Connector(GpuMatrix.from_npa(np.random.rand(dim, k), 'float'))
            true_labels = Connector(GpuMatrix.from_npa(np.array([[0, 1, 0]], np.float32)))
            log_reg = LogisticRegressionCe(init, features, true_labels)
            w_np = log_reg.w.to_host()
            true_labels = true_labels.to_host()

            log_reg.fprop()
            log_reg.bprop()
            dL_dw = log_reg.dL_dw.to_host()
            numerical_grad = np.zeros_like(dL_dw)
            cross_entropy = lambda l, p: -np.sum(l * np.log(p) + (1 - l) * np.log(1 - p))

            epsilon = 1E-2
            for i in xrange(log_reg.w.nrows):
                for j in xrange(log_reg.w.ncols):
                    log_reg.w[i, j] = w_np[i, j] + epsilon
                    log_reg.fprop()
                    probs = log_reg.probs.to_host()
                    plus_cost = cross_entropy(true_labels, probs)

                    log_reg.w[i, j] = w_np[i, j] - epsilon
                    log_reg.fprop()
                    probs = log_reg.probs.to_host()
                    minus_cost = cross_entropy(true_labels, probs)

                    numerical_grad[i, j] = (plus_cost - minus_cost) / (2 * epsilon)
                    log_reg.w[i, j] = w_np[i, j]

            r.append(np.allclose(dL_dw, numerical_grad, atol=1e-3))

        self.assertEqual(sum(r), n)

    def finite_difference_test_features(self):
        r = []
        n = 10

        for i in xrange(n):
            k = 3
            dim = 100

            init = lambda: (np.random.rand(1, dim) * 0.1).astype(np.float32)
            features = Connector(GpuMatrix.from_npa(np.random.rand(dim, k), 'float'), b_usage_context=Context())
            features_np = features.to_host()
            true_labels = Connector(GpuMatrix.from_npa(np.array([[0, 1, 0]], np.float32)))
            log_reg = LogisticRegressionCe(init, features, true_labels)
            true_labels = true_labels.to_host()

            log_reg.fprop()
            log_reg.bprop()
            dL_dfeatures = log_reg.dL_dfeatures.to_host()
            numerical_grad = np.zeros_like(dL_dfeatures)
            cross_entropy = lambda l, p: -np.sum(l * np.log(p) + (1 - l) * np.log(1 - p))

            epsilon = 1E-2
            for i in xrange(features.nrows):
                for j in xrange(features.ncols):
                    features.__setitem__((i, j), features_np[i, j] + epsilon)
                    # features[i, j] = features_np[i, j] + epsilon
                    log_reg.fprop()
                    probs = log_reg.probs.to_host()
                    plus_cost = cross_entropy(true_labels, probs)

                    features.__setitem__((i, j), features_np[i, j] - epsilon)
                    # features[i, j] = features_np[i, j] - epsilon
                    log_reg.fprop()
                    probs = log_reg.probs.to_host()
                    minus_cost = cross_entropy(true_labels, probs)

                    numerical_grad[i, j] = (plus_cost - minus_cost) / (2 * epsilon)
                    features.__setitem__((i, j), features_np[i, j])
                    # log_reg.w[i, j] = features_np[i, j]

            r.append(np.allclose(dL_dfeatures, numerical_grad, atol=1e-3))

        self.assertEqual(sum(r), n)