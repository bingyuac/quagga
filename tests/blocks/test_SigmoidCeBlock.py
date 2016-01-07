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
from quagga.connector import Connector
from quagga.blocks import SigmoidCeBlock


class TestSigmoidCeBlock(TestCase):
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
            batch_size = self.rng.random_integers(2000)
            true_labels = self.rng.randint(2, size=(batch_size, 1)).astype(np.float32)
            mask = (self.rng.rand(batch_size, 1) < 0.8).astype(np.float32)
            x = self.rng.randn(batch_size, 1).astype(np.float32)

            for with_mask in [False, True]:
                quagga.processor_type = 'gpu'
                x_gpu = Connector(Matrix.from_npa(x))
                true_labels_gpu = Connector(Matrix.from_npa(true_labels))
                mask_gpu = Connector(Matrix.from_npa(mask)) if with_mask else None
                sigmoid_ce_block = SigmoidCeBlock(x_gpu, true_labels_gpu, mask_gpu)
                x_gpu.fprop()
                true_labels_gpu.fprop()
                if with_mask:
                    mask_gpu.fprop()
                sigmoid_ce_block.fprop()
                probs_gpu = sigmoid_ce_block.probs.to_host()

                quagga.processor_type = 'cpu'
                x_cpu = Connector(Matrix.from_npa(x))
                true_labels_cpu = Connector(Matrix.from_npa(true_labels))
                mask_cpu = Connector(Matrix.from_npa(mask)) if with_mask else None
                sigmoid_ce_block = SigmoidCeBlock(x_cpu, true_labels_cpu, mask_cpu)
                x_cpu.fprop()
                true_labels_cpu.fprop()
                if with_mask:
                    mask_cpu.fprop()
                sigmoid_ce_block.fprop()
                probs_cpu = sigmoid_ce_block.probs.to_host()

                r.append(np.allclose(probs_gpu, probs_cpu))

        self.assertEqual(sum(r), len(r))

    def test_bprop(self):
        """
        compare `bprop` results for cpu and gpu backends
        """
        r = []
        for i in xrange(self.N):
            batch_size = self.rng.random_integers(2000)
            true_labels = self.rng.randint(2, size=(batch_size, 1)).astype(np.float32)
            mask = (self.rng.rand(batch_size, 1) < 0.8).astype(np.float32)
            x = self.rng.randn(batch_size, 1).astype(np.float32)
            device_id = 0

            for with_mask in [False, True]:
                quagga.processor_type = 'gpu'
                x_gpu = Connector(Matrix.from_npa(x), device_id)
                true_labels_gpu = Connector(Matrix.from_npa(true_labels))
                mask_gpu = Connector(Matrix.from_npa(mask)) if with_mask else None
                sigmoid_ce_block = SigmoidCeBlock(x_gpu, true_labels_gpu, mask_gpu)
                x_gpu.fprop()
                true_labels_gpu.fprop()
                if with_mask:
                    mask_gpu.fprop()
                sigmoid_ce_block.fprop()
                sigmoid_ce_block.bprop()
                dL_dx_gpu = x_gpu.backward_matrix.to_host()

                x_cpu = Connector(Matrix.from_npa(x), device_id)
                true_labels_cpu = Connector(Matrix.from_npa(true_labels))
                mask_cpu = Connector(Matrix.from_npa(mask)) if with_mask else None
                sigmoid_ce_block = SigmoidCeBlock(x_cpu, true_labels_cpu, mask_cpu)
                x_cpu.fprop()
                true_labels_cpu.fprop()
                if with_mask:
                    mask_cpu.fprop()
                sigmoid_ce_block.fprop()
                sigmoid_ce_block.bprop()
                dL_dx_cpu = x_cpu.backward_matrix.to_host()

                r.append(np.allclose(dL_dx_gpu, dL_dx_cpu))

        self.assertEqual(sum(r), len(r))

    def test_theano_grad(self):
        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            batch_size = self.rng.random_integers(2000)
            true_labels = self.rng.randint(2, size=(batch_size, 1)).astype(dtype=np.float32)
            mask = (self.rng.rand(batch_size, 1) < 0.8).astype(np.float32)
            x = self.rng.randn(batch_size, 1).astype(dtype=np.float32)
            device_id = 0

            for with_mask in [False, True]:
                # Theano model
                th_x = T.fmatrix()
                th_mask = T.fmatrix()
                th_true_labels = T.fmatrix()
                if with_mask:
                    probs = T.nnet.sigmoid(th_mask * th_x)
                else:
                    probs = T.nnet.sigmoid(th_x)
                loss = T.mean(T.nnet.binary_crossentropy(probs, th_true_labels))
                if with_mask:
                    get_theano_grads = theano.function([th_x, th_true_labels, th_mask], T.grad(loss, wrt=th_x))
                    th_dL_dx = get_theano_grads(x, true_labels, mask)
                else:
                    get_theano_grads = theano.function([th_x, th_true_labels], T.grad(loss, wrt=th_x))
                    th_dL_dx = get_theano_grads(x, true_labels)

                # quagga model
                x_gpu = Connector(Matrix.from_npa(x), device_id)
                true_labels_gpu = Connector(Matrix.from_npa(true_labels))
                mask_gpu = Connector(Matrix.from_npa(mask)) if with_mask else None
                sigmoid_ce_block = SigmoidCeBlock(x_gpu, true_labels_gpu, mask_gpu)
                x_gpu.fprop()
                true_labels_gpu.fprop()
                if with_mask:
                    mask_gpu.fprop()
                sigmoid_ce_block.fprop()
                sigmoid_ce_block.bprop()
                q_dL_dx = x_gpu.backward_matrix.to_host()

                r.append(np.allclose(th_dL_dx, q_dL_dx))

        self.assertEqual(sum(r), len(r))