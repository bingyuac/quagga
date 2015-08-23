import quagga
import theano
import numpy as np
from unittest import TestCase
from theano import tensor as T
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector
from quagga.blocks import LogisticRegressionCe


class TestLogisticRegressionCe(TestCase):
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
            batch_size = self.rng.random_integers(512)
            features_dim = self.rng.random_integers(2000)
            features = self.rng.rand(batch_size, features_dim).astype(dtype=np.float32)
            true_labels = self.rng.randint(1, size=(batch_size, 1)).astype(dtype=np.float32)

            W_init = self.get_orthogonal_initializer(features_dim, 1)
            b_init = lambda: self.rng.rand(1, 1).astype(dtype=np.float32)

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            features_gpu = Connector(Matrix.from_npa(features))
            true_labels_gpu = Connector(Matrix.from_npa(true_labels))
            lr_gpu = LogisticRegressionCe(W_init, b_init, features_gpu, true_labels_gpu, learning=False)
            lr_gpu.fprop()
            lr_gpu.context.synchronize()
            probs_gpu = lr_gpu.probs.to_host()

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            features_cpu = Connector(Matrix.from_npa(features))
            true_labels_cpu = Connector(Matrix.from_npa(true_labels))
            lr_cpu = LogisticRegressionCe(W_init, b_init, features_cpu, true_labels_cpu, learning=False)
            lr_cpu.fprop()
            lr_cpu.context.synchronize()
            probs_cpu = lr_gpu.probs.to_host()

            r.append(np.allclose(probs_gpu, probs_cpu, rtol=1e-7, atol=1e-3))

        self.assertEqual(sum(r), self.N)

    def test_bprop(self):
        """
        compare `bprop` results for cpu and gpu backends
        """
        r = []
        for i in xrange(self.N):
            batch_size = self.rng.random_integers(512)
            features_dim = self.rng.random_integers(2000)
            features = self.rng.rand(batch_size, features_dim).astype(dtype=np.float32)
            true_labels = self.rng.randint(1, size=(batch_size, 1)).astype(dtype=np.float32)

            W_init = self.get_orthogonal_initializer(features_dim, 1)
            b_init = lambda: self.rng.rand(1, 1).astype(dtype=np.float32)

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            context = Context()
            features_gpu = Connector(Matrix.from_npa(features), context, context)
            true_labels_gpu = Connector(Matrix.from_npa(true_labels))
            lr_gpu = LogisticRegressionCe(W_init, b_init, features_gpu, true_labels_gpu)
            lr_gpu.fprop()
            lr_gpu.bprop()
            lr_gpu.context.synchronize()
            context.synchronize()
            dL_dW_gpu = lr_gpu.dL_dW.to_host()
            dL_db_gpu = lr_gpu.dL_db.to_host()
            dL_dfeatures_gpu = lr_gpu.dL_dfeatures.to_host()

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            context = Context()
            features_cpu = Connector(Matrix.from_npa(features), context, context)
            true_labels_cpu = Connector(Matrix.from_npa(true_labels))
            lr_cpu = LogisticRegressionCe(W_init, b_init, features_cpu, true_labels_cpu)
            lr_cpu.fprop()
            lr_cpu.bprop()
            lr_cpu.context.synchronize()
            context.synchronize()
            dL_dW_cpu = lr_cpu.dL_dW.to_host()
            dL_db_cpu = lr_cpu.dL_db.to_host()
            dL_dfeatures_cpu = lr_cpu.dL_dfeatures.to_host()

            r.append(np.allclose(dL_dW_gpu, dL_dW_cpu, rtol=1e-7, atol=1e-3))
            r.append(np.allclose(dL_db_gpu, dL_db_cpu, rtol=1e-7, atol=1e-3))
            r.append(np.allclose(dL_dfeatures_gpu, dL_dfeatures_cpu, rtol=1e-7, atol=1e-3))

        self.assertEqual(sum(r), self.N * 3)

    def test_theano_grad(self):
        class LogisticRegression(object):
            def __init__(self, W_init, b_init):
                self.W = theano.shared(value=W_init())
                self.b = theano.shared(value=b_init())

            def get_output_expr(self, input_expr):
                return T.nnet.sigmoid(T.dot(input_expr, self.W) + self.b)

        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            batch_size = self.rng.random_integers(512)
            features_dim = self.rng.random_integers(2000)
            features = self.rng.rand(batch_size, features_dim).astype(dtype=np.float32)
            true_labels = self.rng.randint(1, size=(batch_size, 1)).astype(dtype=np.float32)

            W_init = self.get_orthogonal_initializer(features_dim, 1)
            b_init = lambda: self.rng.rand(1, 1).astype(dtype=np.float32)

            # Theano model
            state = self.rng.get_state()
            th_features = T.fmatrix()
            th_true_labels = T.fmatrix()
            lr_layer = LogisticRegression(W_init, lambda: b_init()[0])
            probs = lr_layer.get_output_expr(th_features)
            loss = T.mean(T.nnet.binary_crossentropy(probs, th_true_labels))
            grad_W, grad_b, grad_features = T.grad(loss, wrt=[lr_layer.W, lr_layer.b, th_features])
            get_theano_grads = theano.function([th_features, th_true_labels], [grad_W, grad_b, grad_features])

            # quagga model
            self.rng.set_state(state)
            context = Context()
            features = Connector(Matrix.from_npa(features), context, context)
            true_labels = Connector(Matrix.from_npa(true_labels))
            lr_block = LogisticRegressionCe(W_init, b_init, features, true_labels)
            lr_block.fprop()
            lr_block.bprop()
            lr_block.context.synchronize()
            context.synchronize()

            dL_dW_q = lr_block.dL_dW.to_host()
            dL_db_q = lr_block.dL_db.to_host()
            dL_dfeatures_q = lr_block.dL_dfeatures.to_host()
            dL_dW_th, dL_db_th, dL_dfeatures_th = get_theano_grads(features.to_host(), true_labels.to_host())

            r.append(np.allclose(dL_dW_q, dL_dW_th, rtol=1e-7, atol=1e-3))
            r.append(np.allclose(dL_db_q, dL_db_th, rtol=1e-7, atol=1e-3))
            r.append(np.allclose(dL_dfeatures_q, dL_dfeatures_th, rtol=1e-7, atol=1e-3))

        self.assertEqual(sum(r), self.N * 3)