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
import quagga
import theano
import numpy as np
from unittest import TestCase
from theano import tensor as T
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import SseBlock
from quagga.connector import Connector


class TestSseBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 10

    def test_bprop(self):
        """
        compare `bprop` results for cpu and gpu backends
        """
        r = []
        for i in xrange(self.N):
            batch_size, dim = self.rng.random_integers(2000, size=2)
            y_hat = self.rng.randn(batch_size, dim).astype(dtype=np.float32)
            y = self.rng.randn(batch_size, dim).astype(dtype=np.float32)

            quagga.processor_type = 'gpu'
            context = Context()
            y_hat_gpu = Connector(Matrix.from_npa(y_hat), context, context)
            y_gpu = Connector(Matrix.from_npa(y))
            sse_block = SseBlock(y_hat_gpu, y_gpu)
            sse_block.fprop()
            sse_block.bprop()
            dL_dy_hat_gpu = y_hat_gpu.backward_matrix.to_host()

            quagga.processor_type = 'cpu'
            context = Context()
            y_hat_cpu = Connector(Matrix.from_npa(y_hat), context, context)
            y_cpu = Connector(Matrix.from_npa(y))
            sse_block = SseBlock(y_hat_cpu, y_cpu)
            sse_block.fprop()
            sse_block.bprop()
            dL_dy_hat_cpu = y_hat_cpu.backward_matrix.to_host()

            r.append(np.allclose(dL_dy_hat_gpu, dL_dy_hat_cpu))

        self.assertEqual(sum(r), self.N)

    def test_theano_grad(self):
        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            batch_size, dim = self.rng.random_integers(2000, size=2)
            y_hat = self.rng.randn(batch_size, dim).astype(dtype=np.float32)
            y = self.rng.randn(batch_size, dim).astype(dtype=np.float32)

            # Theano model
            th_y_hat, th_y = T.fmatrix(), T.fmatrix()
            loss = T.mean(T.sum((th_y_hat - th_y) ** 2, axis=1))
            get_theano_grads = theano.function([th_y_hat, th_y], T.grad(loss, wrt=th_y_hat))
            th_dL_dy_hat = get_theano_grads(y_hat, y)

            # quagga model
            context = Context()
            y_hat_gpu = Connector(Matrix.from_npa(y_hat), context, context)
            y_gpu = Connector(Matrix.from_npa(y))
            sigmoid_ce_block = SseBlock(y_hat_gpu, y_gpu)
            sigmoid_ce_block.fprop()
            sigmoid_ce_block.bprop()
            q_dL_dy_hat = y_hat_gpu.backward_matrix.to_host()

            r.append(np.allclose(th_dL_dy_hat, q_dL_dy_hat))

        self.assertEqual(sum(r), self.N)