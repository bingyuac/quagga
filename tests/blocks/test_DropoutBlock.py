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
from itertools import izip
from unittest import TestCase
from theano import tensor as T
from quagga.matrix import Matrix
from quagga.blocks import DotBlock
from quagga.connector import Connector
from quagga.blocks import DropoutBlock
from quagga.blocks import SigmoidCeBlock


class TestDropoutBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 20

    def test_theano_grad(self):
        class LogisticRegressionLayer(object):
            def __init__(self, W, b):
                self.W = theano.shared(value=W)
                self.b = theano.shared(value=b[0])

            def get_output_expr(self, input_expr):
                return T.nnet.sigmoid(T.dot(input_expr, self.W) + self.b)

        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            batch_size, x_dim = self.rng.random_integers(3000, size=2)
            x = self.rng.rand(batch_size, x_dim).astype(np.float32)
            lr_dot_W = self.rng.rand(x_dim, 1).astype(np.float32)
            lr_dot_b = self.rng.rand(1, 1).astype(np.float32)
            true_labels = self.rng.randint(2, size=(batch_size, 1)).astype(np.float32)
            dropout_prob = self.rng.uniform()
            seed = self.rng.randint(1000)
            device_id = 0

            # quagga model
            state = self.rng.get_state()
            x_gpu = Connector(Matrix.from_npa(x), device_id)
            true_labels_gpu = Connector(Matrix.from_npa(true_labels))
            lr_dot_W_gpu = Connector(Matrix.from_npa(lr_dot_W), device_id)
            lr_dot_b_gpu = Connector(Matrix.from_npa(lr_dot_b), device_id)

            dropout_block = DropoutBlock(x_gpu, dropout_prob, seed)
            lrdot_block = DotBlock(lr_dot_W_gpu, lr_dot_b_gpu, dropout_block.output)
            sce_block = SigmoidCeBlock(lrdot_block.output, true_labels_gpu)
            x_gpu.fprop()
            true_labels_gpu.fprop()
            lr_dot_W_gpu.fprop()
            lr_dot_b_gpu.fprop()
            dropout_block.fprop()
            lrdot_block.fprop()
            sce_block.fprop()
            sce_block.bprop()
            lrdot_block.bprop()
            dropout_block.bprop()
            q_grads = [lr_dot_W_gpu.backward_matrix.to_host(),
                       lr_dot_b_gpu.backward_matrix.to_host(),
                       x_gpu.backward_matrix.to_host()]
            mask = (dropout_block.output.to_host() != 0).astype(np.float32)

            # Theano model
            self.rng.set_state(state)
            th_x = T.fmatrix()
            th_true_labels = T.fmatrix()
            lr_layer = LogisticRegressionLayer(lr_dot_W, lr_dot_b)
            probs = lr_layer.get_output_expr(th_x * mask)
            loss = T.mean(T.nnet.binary_crossentropy(probs, th_true_labels))
            th_grads = T.grad(loss, wrt=[lr_layer.W, lr_layer.b, th_x])
            get_theano_grads = theano.function([th_x, th_true_labels], th_grads)
            th_grads = get_theano_grads(x, true_labels)

            for i, (q_grad, th_grad) in enumerate(izip(q_grads, th_grads)):
                r.append(np.allclose(q_grad, th_grad))

        self.assertEqual(sum(r), len(r))