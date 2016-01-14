# ----------------------------------------------------------------------------
# Copyright 2015 Grammarly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from itertools import izip
from unittest import TestCase

import theano
import numpy as np
from theano import tensor as T

import quagga
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import DotBlock
from quagga.connector import Connector
from quagga.utils import List
from quagga.blocks import SigmoidCeBlock
from quagga.blocks import SequentialMeanPoolingBlock
from quagga.blocks import SequentialHorizontalStackBlock


class TestSequentialHorizontalStackBlock(TestCase):
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
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            batch_size = self.rng.random_integers(512)
            dim_x, dim_y = self.rng.random_integers(1500, size=2)
            x = [self.rng.rand(batch_size, dim_x).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]
            y = [self.rng.rand(batch_size, dim_y).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            x_gpu = List([Connector(Matrix.from_npa(e)) for e in x])
            y_gpu = List([Connector(Matrix.from_npa(e)) for e in y])
            seq_hstack_block_gpu = SequentialHorizontalStackBlock(x_gpu, y_gpu)
            x_gpu.length = sequence_len
            y_gpu.length = sequence_len
            if sequence_len == 0:
                pass
            seq_hstack_block_gpu.fprop()
            output_sequence_gpu = seq_hstack_block_gpu.output.to_host()

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            x_cpu = List([Connector(Matrix.from_npa(e)) for e in x])
            y_cpu = List([Connector(Matrix.from_npa(e)) for e in y])
            seq_hstack_block_cpu = SequentialHorizontalStackBlock(x_cpu, y_cpu)
            x_cpu.length = sequence_len
            y_cpu.length = sequence_len
            seq_hstack_block_cpu.fprop()
            output_sequence_cpu = seq_hstack_block_cpu.output.to_host()

            for out_gpu, out_cpu in izip(output_sequence_gpu, output_sequence_cpu):
                if not np.allclose(out_gpu, out_cpu):
                    r.append(False)
                    break
            else:
                r.append(True)

        self.assertEqual(sum(r), self.N)

    def test_bprop(self):
        """
        compare `bprop` results for cpu and gpu backends
        """
        device_id = 0
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            batch_size = self.rng.random_integers(256)
            dim_x, dim_y = self.rng.random_integers(1280, size=2)
            x = [self.rng.rand(batch_size, dim_x).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]
            y = [self.rng.rand(batch_size, dim_y).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]

            state = self.rng.get_state()
            quagga.processor_type = 'gpu'
            context = Context()
            x_gpu = List([Connector(Matrix.from_npa(e), device_id) for e in x])
            y_gpu = List([Connector(Matrix.from_npa(e), device_id) for e in y])
            seq_hstack_block_gpu = SequentialHorizontalStackBlock(x_gpu, y_gpu)
            x_gpu.length = sequence_len
            y_gpu.length = sequence_len
            _, dL_doutput_sequence = izip(*seq_hstack_block_gpu.output.register_usage(device_id, device_id))
            seq_hstack_block_gpu.fprop()
            for dL_doutput in dL_doutput_sequence:
                random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
                dL_doutput.assign(context, Matrix.from_npa(random_matrix, 'float'))
            seq_hstack_block_gpu.bprop()
            dL_dx_matrices_gpu = [e.backward_matrix.to_host() for e in x_gpu]
            dL_dy_matrices_gpu = [e.backward_matrix.to_host() for e in y_gpu]

            self.rng.set_state(state)
            quagga.processor_type = 'cpu'
            context = Context()
            x_cpu = List([Connector(Matrix.from_npa(e), device_id) for e in x])
            y_cpu = List([Connector(Matrix.from_npa(e), device_id) for e in y])
            seq_hstack_block_cpu = SequentialHorizontalStackBlock(x_cpu, y_cpu)
            x_cpu.length = sequence_len
            y_cpu.length = sequence_len
            _, dL_doutput_sequence = izip(*seq_hstack_block_cpu.output.register_usage(device_id, device_id))
            seq_hstack_block_cpu.fprop()
            for dL_doutput in dL_doutput_sequence:
                random_matrix = self.rng.rand(dL_doutput.nrows, dL_doutput.ncols)
                dL_doutput.assign(context, Matrix.from_npa(random_matrix, 'float'))
            seq_hstack_block_cpu.bprop()
            dL_dx_matrices_cpu = [e.backward_matrix.to_host() for e in x_cpu]
            dL_dy_matrices_cpu = [e.backward_matrix.to_host() for e in y_cpu]

            for dL_dx_gpu, dL_dx_cpu in izip(dL_dx_matrices_gpu, dL_dx_matrices_cpu):
                if not np.allclose(dL_dx_gpu, dL_dx_cpu):
                    r.append(False)
                    break
            else:
                r.append(True)
            for dL_dy_gpu, dL_dy_cpu in izip(dL_dy_matrices_gpu, dL_dy_matrices_cpu):
                if not np.allclose(dL_dy_gpu, dL_dy_cpu):
                    r.append(False)
                    break
            else:
                r.append(True)

            del x_gpu
            del y_gpu
            del seq_hstack_block_gpu
            del dL_dx_matrices_gpu
            del dL_dy_matrices_gpu

        self.assertEqual(sum(r), self.N * 2)

    def test_theano_grad(self):
        device_id = 0
        class SequentialHorizontalStackLayer(object):
            def get_output_expr(self, x_sequence, y_sequence):
                return T.concatenate((x_sequence, y_sequence), axis=1)

        class SequentialMeanPoolingLayer(object):
            def get_output_expr(self, input_sequence):
                return T.mean(input_sequence, axis=2)

        class LogisticRegressionLayer(object):
            def __init__(self, W_init, b_init):
                self.W = theano.shared(value=W_init())
                self.b = theano.shared(value=b_init())

            def get_output_expr(self, input_expr):
                return T.nnet.sigmoid(T.dot(input_expr, self.W) + self.b)

        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            batch_size = self.rng.random_integers(256)
            dim_x, dim_y = self.rng.random_integers(1280, size=2)
            x = [self.rng.rand(batch_size, dim_x).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]
            y = [self.rng.rand(batch_size, dim_y).astype(dtype=np.float32) for _ in xrange(max_input_sequence_len)]
            true_labels = self.rng.randint(1, size=(batch_size, 1)).astype(dtype=np.float32)

            W_init = self.get_orthogonal_initializer(dim_x + dim_y, 1)
            b_init = lambda: self.rng.rand(1, 1).astype(dtype=np.float32)

            # Theano model
            state = self.rng.get_state()
            th_x = T.ftensor3()
            th_y = T.ftensor3()
            th_true_labels = T.fmatrix()
            shs_layer = SequentialHorizontalStackLayer()
            smp_layer = SequentialMeanPoolingLayer()
            lr_layer = LogisticRegressionLayer(W_init, lambda: b_init()[0])
            probs = shs_layer.get_output_expr(th_x, th_y)
            probs = lr_layer.get_output_expr(smp_layer.get_output_expr(probs))
            loss = T.mean(T.nnet.binary_crossentropy(probs, th_true_labels))
            grads = T.grad(loss, wrt=[th_x, th_y])
            get_grads = theano.function([th_x, th_y, th_true_labels], grads)
            dL_dx_sequence_th, dL_dy_sequence_th = get_grads(np.dstack(x[:sequence_len]), np.dstack(y[:sequence_len]), true_labels)

            # quagga model
            self.rng.set_state(state)
            W = Connector(Matrix.from_npa(W_init(), device_id=device_id), device_id)
            b = Connector(Matrix.from_npa(b_init(), device_id=device_id), device_id)
            x = List([Connector(Matrix.from_npa(e), device_id) for e in x])
            y = List([Connector(Matrix.from_npa(e), device_id) for e in y])
            true_labels = Connector(Matrix.from_npa(true_labels))
            shs_block = SequentialHorizontalStackBlock(x, y)
            smp_block = SequentialMeanPoolingBlock(shs_block.output)
            dot_block = DotBlock(W, b, smp_block.output)
            sce_block = SigmoidCeBlock(dot_block.output, true_labels)
            x.length = sequence_len
            y.length = sequence_len
            shs_block.fprop()
            smp_block.fprop()
            dot_block.fprop()
            sce_block.fprop()
            sce_block.bprop()
            dot_block.bprop()
            smp_block.bprop()
            shs_block.bprop()
            dL_dx_sequence = [e.backward_matrix.to_host() for e in x]
            dL_dy_sequence = [e.backward_matrix.to_host() for e in y]

            for i in xrange(dL_dx_sequence_th.shape[-1]):
                if not np.allclose(dL_dx_sequence[i], dL_dx_sequence_th[..., i], atol=1.e-6):
                    r.append(False)
                    break
            else:
                r.append(True)
            for i in xrange(dL_dy_sequence_th.shape[-1]):
                if not np.allclose(dL_dy_sequence[i], dL_dy_sequence_th[..., i], atol=1.e-6):
                    r.append(False)
                    break
            else:
                r.append(True)

        self.assertEqual(sum(r), self.N * 2)