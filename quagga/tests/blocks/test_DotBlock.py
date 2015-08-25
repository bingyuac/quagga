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
from quagga.blocks import SigmoidCeBlock


class TestDotBlock(TestCase):
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
            batch_size, x_dim, output_dim = self.rng.random_integers(2000, size=3)
            x = self.rng.rand(batch_size, x_dim).astype(dtype=np.float32)
            W_init = self.get_orthogonal_initializer(x_dim, output_dim)
            b_init = (lambda: self.rng.rand(1, output_dim).astype(dtype=np.float32)) if self.rng.randint(2) else None

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            x_gpu = Connector(Matrix.from_npa(x))
            dot_block_gpu = DotBlock(W_init, b_init, x_gpu, learning=False)
            dot_block_gpu.fprop()
            cudart.cuda_device_synchronize()
            output_gpu = dot_block_gpu.output.to_host()

            quagga.processor_type = 'cpu'
            self.rng.set_state(state)
            x_cpu = Connector(Matrix.from_npa(x))
            dot_block_cpu = DotBlock(W_init, b_init, x_cpu, learning=False)
            dot_block_cpu.fprop()
            cudart.cuda_device_synchronize()
            output_cpu = dot_block_cpu.output.to_host()

            r.append(np.allclose(output_gpu, output_cpu, atol=1e-5))

        self.assertEqual(sum(r), self.N)

    def test_bprop(self):
        """
        compare `bprop` results for cpu and gpu backends
        """
        n = 0
        r = []
        for i in xrange(self.N):
            batch_size, x_dim, output_dim = self.rng.random_integers(2000, size=3)
            x = self.rng.rand(batch_size, x_dim).astype(dtype=np.float32)
            W_init = self.get_orthogonal_initializer(x_dim, output_dim)
            b_init = (lambda: self.rng.rand(1, output_dim).astype(dtype=np.float32)) if self.rng.randint(2) else None

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            context = Context()
            x_gpu = Connector(Matrix.from_npa(x), context, context)
            dot_block_gpu = DotBlock(W_init, b_init, x_gpu)
            dot_block_gpu.fprop()
            _, dL_doutput = dot_block_gpu.output.register_usage(context, context)
            random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
            Matrix.from_npa(random_matrix, 'float').copy(context, dL_doutput)
            dot_block_gpu.bprop()
            cudart.cuda_device_synchronize()
            dL_dx_gpu = x_gpu.backward_matrix.to_host()
            dL_dW_gpu = dot_block_gpu.dL_dW.to_host()
            if b_init:
                dL_db_gpu = dot_block_gpu.dL_db.to_host()

            quagga.processor_type = 'cpu'
            self.rng.set_state(state)
            context = Context()
            x_cpu = Connector(Matrix.from_npa(x), context, context)
            dot_block_cpu = DotBlock(W_init, b_init, x_cpu)
            dot_block_cpu.fprop()
            _, dL_doutput = dot_block_cpu.output.register_usage(context, context)
            random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
            Matrix.from_npa(random_matrix, 'float').copy(context, dL_doutput)
            dot_block_cpu.bprop()
            dL_dx_cpu = x_cpu.backward_matrix.to_host()
            dL_dW_cpu = dot_block_cpu.dL_dW.to_host()
            if b_init:
                dL_db_cpu = dot_block_cpu.dL_db.to_host()

            r.append(np.allclose(dL_dx_gpu, dL_dx_cpu, atol=1e-5))
            r.append(np.allclose(dL_dW_gpu, dL_dW_cpu, atol=1e-5))
            if b_init:
                r.append(np.allclose(dL_db_gpu, dL_db_cpu, atol=1e-5))
                n += 1
            n += 2

        self.assertEqual(sum(r), n)

    def test_theano_grad(self):
        class DotLayer(object):
            def __init__(self, W_init, b_init):
                self.W = theano.shared(value=W_init())
                if b_init:
                    self.b = theano.shared(value=b_init()[0])

            def get_output_expr(self, input_expr):
                if hasattr(self, 'b'):
                    return T.dot(input_expr, self.W) + self.b
                else:
                    return T.dot(input_expr, self.W)

        class LogisticRegressionLayer(object):
            def __init__(self, W_init, b_init):
                self.W = theano.shared(value=W_init())
                if b_init:
                    self.b = theano.shared(value=b_init()[0])

            def get_output_expr(self, input_expr):
                if hasattr(self, 'b'):
                    return T.nnet.sigmoid(T.dot(input_expr, self.W) + self.b)
                else:
                    return T.nnet.sigmoid(T.dot(input_expr, self.W))

        quagga.processor_type = 'gpu'
        n = 0
        r = []
        for i in xrange(self.N):
            batch_size, x_dim, output_dim = self.rng.random_integers(2000, size=3)
            x = self.rng.rand(batch_size, x_dim).astype(dtype=np.float32)
            dot_W_init = self.get_orthogonal_initializer(x_dim, output_dim)
            dot_b_init = (lambda: self.rng.rand(1, output_dim).astype(dtype=np.float32)) if self.rng.randint(2) else None
            lrdot_W_init = self.get_orthogonal_initializer(output_dim, 1)
            lrdot_b_init = (lambda: self.rng.rand(1, 1).astype(dtype=np.float32)) if self.rng.randint(2) else None
            true_labels = self.rng.randint(2, size=(batch_size, 1)).astype(dtype=np.float32)

            # Theano model
            state = self.rng.get_state()
            th_x = T.fmatrix()
            th_true_labels = T.fmatrix()
            dot_layer = DotLayer(dot_W_init, dot_b_init)
            lr_layer = LogisticRegressionLayer(lrdot_W_init, lrdot_b_init)
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
            context = Context()
            x = Connector(Matrix.from_npa(x), context, context)
            true_labels = Connector(Matrix.from_npa(true_labels))
            dot_block = DotBlock(dot_W_init, dot_b_init, x)
            lrdot_block = DotBlock(lrdot_W_init, lrdot_b_init, dot_block.output)
            sce_block = SigmoidCeBlock(lrdot_block.output, true_labels)
            dot_block.fprop()
            lrdot_block.fprop()
            sce_block.fprop()
            sce_block.bprop()
            lrdot_block.bprop()
            dot_block.bprop()
            cudart.cuda_device_synchronize()
            q_grads = [lrdot_block.dL_dW.to_host(),
                       dot_block.dL_dW.to_host(),
                       x.backward_matrix.to_host()]
            if hasattr(lrdot_block, 'b'):
                q_grads.append(lrdot_block.dL_db.to_host())
            if hasattr(dot_block, 'b'):
                q_grads.append(dot_block.dL_db.to_host())

            for th_grad, q_grad in izip(th_grads, q_grads):
                r.append(np.allclose(th_grad, q_grad, atol=1e-7))
            n += len(q_grads)

        self.assertEqual(sum(r), n)