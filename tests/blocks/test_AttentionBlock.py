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
import theano
import numpy as np
from itertools import izip
from unittest import TestCase
from quagga.utils import List
from theano import tensor as T
from quagga.matrix import Matrix
from quagga.blocks import DotBlock
from quagga.connector import Connector
from quagga.blocks import AttentionBlock
from quagga.blocks import SigmoidCeBlock
from quagga.utils.initializers import Orthogonal


class TestAttentionBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 50

    @classmethod
    def get_orthogonal_matrix(cls, nrows, ncols):
        return Orthogonal(nrows, ncols)()

    def test_theano_grad(self):
        class AttentionLayer(object):
            def __init__(self, u, mask=None):
                self.u = theano.shared(value=u)
                self.mask = mask

            def get_output_expr(self, input_expr):
                input_expr = input_expr.dimshuffle(0, 2, 1)
                pre_a = T.dot(input_expr, self.u)[:, :, 0]
                if self.mask:
                    pre_a = self.mask * pre_a - \
                            (1 - self.mask) * 3.402823466e+38
                a = T.nnet.softmax(pre_a)[:, :, np.newaxis]
                return T.sum(a * input_expr, axis=1)

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

        r = []
        for i in xrange(self.N):
            batch_size = self.rng.random_integers(500)
            x_dim = self.rng.random_integers(3000)
            n_ts = self.rng.random_integers(100)
            x = [self.rng.rand(batch_size, x_dim).astype(np.float32) for _ in xrange(n_ts)]
            u = self.get_orthogonal_matrix(x_dim, 1)
            lr_dot_W = self.get_orthogonal_matrix(x_dim, 1)
            lr_dot_b = self.rng.rand(1, 1).astype(np.float32) if self.rng.randint(2) else None
            true_labels = self.rng.randint(2, size=(batch_size, 1)).astype(np.float32)
            mask = self.rng.randint(2, size=(batch_size, n_ts)).astype(np.float32) if self.rng.randint(2) else None
            device_id = 0

            # Theano model
            state = self.rng.get_state()
            th_x = T.ftensor3()
            th_mask = T.fmatrix() if mask is not None else None

            th_true_labels = T.fmatrix()
            attnt_layer = AttentionLayer(u, th_mask)
            lr_layer = LogisticRegressionLayer(lr_dot_W, lr_dot_b)
            probs = th_x
            for layer in [attnt_layer, lr_layer]:
                probs = layer.get_output_expr(probs)
            loss = T.mean(T.nnet.binary_crossentropy(probs, th_true_labels))

            params = [lr_layer.W, attnt_layer.u, th_x]
            if hasattr(lr_layer, 'b'):
                params.append(lr_layer.b)
            th_grads = T.grad(loss, wrt=params)
            get_theano_grads = theano.function([th_x, th_true_labels] + ([th_mask] if mask is not None else []), th_grads)
            th_grads = get_theano_grads(*([np.dstack(x), true_labels] + ([mask] if mask is not None else [])))

            # quagga model
            self.rng.set_state(state)
            x = List([Connector(Matrix.from_npa(e), device_id) for e in x])
            u = Connector(Matrix.from_npa(u), device_id)
            lr_dot_W = Connector(Matrix.from_npa(lr_dot_W), device_id)
            lr_dot_b = Connector(Matrix.from_npa(lr_dot_b), device_id) if lr_dot_b is not None else lr_dot_b
            true_labels = Connector(Matrix.from_npa(true_labels))
            if mask is not None:
                mask = Connector(Matrix.from_npa(mask))

            attnt_block = AttentionBlock(x, u, mask)
            lrdot_block = DotBlock(lr_dot_W, lr_dot_b, attnt_block.output)
            sce_block = SigmoidCeBlock(lrdot_block.output, true_labels)

            x.fprop()
            true_labels.fprop()
            u.fprop()
            lr_dot_W.fprop()
            if lr_dot_b:
                lr_dot_b.fprop()
            attnt_block.fprop()
            lrdot_block.fprop()
            sce_block.fprop()
            sce_block.bprop()
            lrdot_block.bprop()
            attnt_block.bprop()
            q_grads = [lr_dot_W.backward_matrix.to_host(),
                       u.backward_matrix.to_host(),
                       np.dstack([e.backward_matrix.to_host() for e in x])]
            if lr_dot_b:
                q_grads.append(lr_dot_b.backward_matrix.to_host())

            for th_grad, q_grad in izip(th_grads, q_grads):
                r.append(np.allclose(th_grad, q_grad, atol=1.e-7))
                print r[-1]

        self.assertEqual(sum(r), len(r))