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


class TestDotBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 15

    @classmethod
    def get_orthogonal_matrix(cls, nrows, ncols):
        shape = (nrows, ncols)
        a = cls.rng.normal(0.0, 1.0, shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == shape else v
        q = q.reshape(shape).astype(np.float32)
        return q

    def test_fprop(self):
        """
        compare `fprop` results for cpu and gpu backends
        """
        r = []
        for i in xrange(self.N):
            batch_size, x_dim, output_dim = self.rng.random_integers(2000, size=3)
            x = self.rng.rand(batch_size, x_dim).astype(dtype=np.float32)
            W = self.get_orthogonal_matrix(x_dim, output_dim)
            b = self.rng.rand(1, output_dim).astype(dtype=np.float32) if self.rng.randint(2) else None

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            x_gpu = Connector(Matrix.from_npa(x))
            W_gpu = Connector(Matrix.from_npa(W))
            b_gpu = Connector(Matrix.from_npa(b)) if b is not None else b
            dot_block_gpu = DotBlock(W_gpu, b_gpu, x_gpu, learning=False)
            x_gpu.fprop()
            W_gpu.fprop()
            if b_gpu:
                b_gpu.fprop()
            dot_block_gpu.fprop()
            output_gpu = dot_block_gpu.output.to_host()

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            x_cpu = Connector(Matrix.from_npa(x))
            W_cpu = Connector(Matrix.from_npa(W))
            b_cpu = Connector(Matrix.from_npa(b)) if b is not None else b
            dot_block_cpu = DotBlock(W_cpu, b_cpu, x_cpu, learning=False)
            x_cpu.fprop()
            W_cpu.fprop()
            if b_cpu:
                b_cpu.fprop()
            dot_block_cpu.fprop()
            output_cpu = dot_block_cpu.output.to_host()

            r.append(np.allclose(output_gpu, output_cpu, atol=1e-5))

        self.assertEqual(sum(r), self.N)

    def test_bprop(self):
        """
        compare `bprop` results for cpu and gpu backends
        """
        r = []
        for i in xrange(self.N):
            batch_size, x_dim, output_dim = self.rng.random_integers(2000, size=3)
            x = self.rng.rand(batch_size, x_dim).astype(dtype=np.float32)
            W = self.get_orthogonal_matrix(x_dim, output_dim)
            b = self.rng.rand(1, output_dim).astype(dtype=np.float32) if self.rng.randint(2) else None
            device_id = 0

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            context = Context()
            x_gpu = Connector(Matrix.from_npa(x), device_id)
            W_gpu = Connector(Matrix.from_npa(W), device_id)
            b_gpu = Connector(Matrix.from_npa(b), device_id) if b is not None else b
            dot_block_gpu = DotBlock(W_gpu, b_gpu, x_gpu)
            x_gpu.fprop()
            W_gpu.fprop()
            if b_gpu:
                b_gpu.fprop()
            dot_block_gpu.fprop()
            _, dL_doutput = dot_block_gpu.output.register_usage(device_id, device_id)
            random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
            dL_doutput.assign(context, Matrix.from_npa(random_matrix, 'float'))
            dot_block_gpu.bprop()
            if b is not None:
                dL_db_gpu = b_gpu.backward_matrix.to_host()
            dL_dW_gpu = W_gpu.backward_matrix.to_host()
            dL_dx_gpu = x_gpu.backward_matrix.to_host()

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            context = Context()
            x_cpu = Connector(Matrix.from_npa(x), device_id)
            W_cpu = Connector(Matrix.from_npa(W), device_id)
            b_cpu = Connector(Matrix.from_npa(b), device_id) if b is not None else b
            dot_block_cpu = DotBlock(W_cpu, b_cpu, x_cpu)
            x_cpu.fprop()
            W_cpu.fprop()
            if b_cpu:
                b_cpu.fprop()
            dot_block_cpu.fprop()
            _, dL_doutput = dot_block_cpu.output.register_usage(device_id, device_id)
            random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
            dL_doutput.assign(context, Matrix.from_npa(random_matrix, 'float'))
            dot_block_cpu.bprop()
            if b is not None:
                dL_db_cpu = b_cpu.backward_matrix.to_host()
            dL_dW_cpu = W_cpu.backward_matrix.to_host()
            dL_dx_cpu = x_cpu.backward_matrix.to_host()

            r.append(np.allclose(dL_dx_gpu, dL_dx_cpu, atol=1e-5))
            r.append(np.allclose(dL_dW_gpu, dL_dW_cpu, atol=1e-5))
            if b is not None:
                r.append(np.allclose(dL_db_gpu, dL_db_cpu, atol=1e-5))

        self.assertEqual(sum(r), len(r))

    def test_theano_grad(self):
        class DotLayer(object):
            def __init__(self, W, b):
                self.W = theano.shared(value=W)
                if b is not None:
                    self.b = theano.shared(value=b[0])

            def get_output_expr(self, input_expr):
                if hasattr(self, 'b'):
                    return T.dot(input_expr, self.W) + self.b
                else:
                    return T.dot(input_expr, self.W)

        class LogisticRegressionLayer(object):
            def __init__(self, W, b):
                self.W = theano.shared(value=W)
                if b is not None:
                    self.b = theano.shared(value=b[0])

            def get_output_expr(self, input_expr):
                if hasattr(self, 'b'):
                    return T.nnet.sigmoid(T.dot(input_expr, self.W) + self.b)
                else:
                    return T.nnet.sigmoid(T.dot(input_expr, self.W))

        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            batch_size, x_dim, output_dim = self.rng.random_integers(2000, size=3)
            x = self.rng.rand(batch_size, x_dim).astype(dtype=np.float32)
            dot_W = self.get_orthogonal_matrix(x_dim, output_dim)
            dot_b = self.rng.rand(1, output_dim).astype(dtype=np.float32) if self.rng.randint(2) else None
            lr_dot_W = self.get_orthogonal_matrix(output_dim, 1)
            lr_dot_b = self.rng.rand(1, 1).astype(dtype=np.float32) if self.rng.randint(2) else None
            true_labels = self.rng.randint(2, size=(batch_size, 1)).astype(dtype=np.float32)
            device_id = 0

            # Theano model
            state = self.rng.get_state()
            th_x = T.fmatrix()
            th_true_labels = T.fmatrix()
            dot_layer = DotLayer(dot_W, dot_b)
            lr_layer = LogisticRegressionLayer(lr_dot_W, lr_dot_b)
            probs = th_x
            for layer in [dot_layer, lr_layer]:
                probs = layer.get_output_expr(probs)
            loss = T.mean(T.nnet.binary_crossentropy(probs, th_true_labels))

            params = [lr_layer.W, dot_layer.W, th_x]
            if hasattr(lr_layer, 'b'):
                params.append(lr_layer.b)
            if hasattr(dot_layer, 'b'):
                params.append(dot_layer.b)
            th_grads = T.grad(loss, wrt=params)
            get_theano_grads = theano.function([th_x, th_true_labels], th_grads)
            th_grads = get_theano_grads(x, true_labels)

            # quagga model
            self.rng.set_state(state)
            x = Connector(Matrix.from_npa(x), device_id)
            true_labels = Connector(Matrix.from_npa(true_labels))
            dot_W = Connector(Matrix.from_npa(dot_W), device_id)
            dot_b = Connector(Matrix.from_npa(dot_b), device_id) if dot_b is not None else dot_b
            lr_dot_W = Connector(Matrix.from_npa(lr_dot_W), device_id)
            lr_dot_b = Connector(Matrix.from_npa(lr_dot_b), device_id) if lr_dot_b is not None else lr_dot_b

            dot_block = DotBlock(dot_W, dot_b, x)
            lrdot_block = DotBlock(lr_dot_W, lr_dot_b, dot_block.output)
            sce_block = SigmoidCeBlock(lrdot_block.output, true_labels)
            x.fprop()
            true_labels.fprop()
            dot_W.fprop()
            if dot_b:
                dot_b.fprop()
            lr_dot_W.fprop()
            if lr_dot_b:
                lr_dot_b.fprop()
            dot_block.fprop()
            lrdot_block.fprop()
            sce_block.fprop()
            sce_block.bprop()
            lrdot_block.bprop()
            dot_block.bprop()
            q_grads = [lr_dot_W.backward_matrix.to_host(),
                       dot_W.backward_matrix.to_host(),
                       x.backward_matrix.to_host()]
            if lr_dot_b:
                q_grads.append(lr_dot_b.backward_matrix.to_host())
            if dot_b:
                q_grads.append(dot_b.backward_matrix.to_host())

            for th_grad, q_grad in izip(th_grads, q_grads):
                r.append(np.allclose(th_grad, q_grad))

        self.assertEqual(sum(r), len(r))