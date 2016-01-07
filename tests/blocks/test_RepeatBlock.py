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
from quagga.blocks import RepeatBlock
from quagga.connector import Connector
from quagga.blocks import SoftmaxCeBlock


class TestRepeatBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 15

    @classmethod
    def get_normal_matrix(cls, nrows, ncols):
        return cls.rng.normal(0.0, 1.0, (nrows, ncols)).astype(np.float32)

    def test_fprop(self):
        r = []
        for i in xrange(self.N):
            repeats = self.rng.random_integers(42)
            axis = self.rng.randint(2)
            input_dim, output_dim = self.rng.random_integers(2000, size=2)
            x = self.get_normal_matrix(input_dim, output_dim)

            output = {}
            for processor_type in ['gpu', 'cpu']:
                quagga.processor_type = processor_type
                qx = Connector(Matrix.from_npa(x))
                repeat_block = RepeatBlock(qx, repeats, axis)
                qx.fprop()
                repeat_block.fprop()
                output[processor_type] = repeat_block.output.to_host()

            r.append(np.allclose(output['gpu'], output['cpu']))

        self.assertEqual(sum(r), len(r))

    def test_bprop(self):
        r = []
        for i in xrange(self.N):
            repeats = self.rng.random_integers(42)
            axis = self.rng.randint(2)
            input_dim, output_dim = self.rng.random_integers(2000, size=2)
            x = self.get_normal_matrix(input_dim, output_dim)
            input_dim = input_dim if axis else input_dim * repeats
            true_labels = self.rng.randint(output_dim, size=(input_dim, 1)).astype(np.int32)
            device_id = 0

            output = {}
            for processor_type in ['gpu', 'cpu']:
                quagga.processor_type = processor_type
                qx = Connector(Matrix.from_npa(x), device_id)
                qtrue_labels = Connector(Matrix.from_npa(true_labels))
                repeat_block = RepeatBlock(qx, repeats, axis)
                sce_block = SoftmaxCeBlock(repeat_block.output, qtrue_labels)
                qx.fprop()
                qtrue_labels.fprop()
                repeat_block.fprop()
                sce_block.fprop()
                sce_block.bprop()
                repeat_block.bprop()
                output[processor_type] = qx.backward_matrix.to_host()

            r.append(np.allclose(output['gpu'], output['cpu']))

        self.assertEqual(sum(r), len(r))

    def test_theano_fprop(self):
        r = []
        for i in xrange(self.N):
            repeats = self.rng.random_integers(42)
            axis = self.rng.randint(2)
            input_dim, output_dim = self.rng.random_integers(2000, size=2)
            x = self.get_normal_matrix(input_dim, output_dim)

            quagga.processor_type = 'gpu'
            qx = Connector(Matrix.from_npa(x))
            repeat_block = RepeatBlock(qx, repeats, axis)
            qx.fprop()
            repeat_block.fprop()
            qoutput = repeat_block.output.to_host()

            th_x = T.fmatrix()
            reps = [1, 1]
            reps[axis] = repeats
            th_output = T.tile(th_x, reps)
            th_output = theano.function([th_x], th_output)(x)

            r.append(np.allclose(qoutput, th_output))

        self.assertEqual(sum(r), len(r))

    def test_theano_bprop(self):
        r = []
        for i in xrange(self.N):
            repeats = self.rng.random_integers(42)
            axis = self.rng.randint(2)
            input_dim, output_dim = self.rng.random_integers(2000, size=2)
            x = self.get_normal_matrix(input_dim, output_dim)
            input_dim = input_dim if axis else input_dim * repeats
            true_labels = self.rng.randint(output_dim, size=(input_dim, 1)).astype(np.int32)
            device_id = 0

            quagga.processor_type = 'gpu'
            qx = Connector(Matrix.from_npa(x), device_id)
            qtrue_labels = Connector(Matrix.from_npa(true_labels))
            repeat_block = RepeatBlock(qx, repeats, axis)
            sce_block = SoftmaxCeBlock(repeat_block.output, qtrue_labels)
            qx.fprop()
            qtrue_labels.fprop()
            repeat_block.fprop()
            sce_block.fprop()
            sce_block.bprop()
            repeat_block.bprop()
            q_dL_dx = qx.backward_matrix.to_host()

            th_x = T.fmatrix()
            th_true_labels = T.ivector()
            reps = [1, 1]
            reps[axis] = repeats
            th_output = T.tile(th_x, reps)
            th_output = T.nnet.softmax(th_output)
            loss = T.mean(T.nnet.categorical_crossentropy(th_output, th_true_labels))
            get_grads = theano.function([th_x, th_true_labels], T.grad(loss, th_x))
            th_dL_dx = get_grads(x, true_labels[:, 0])

            r.append(np.allclose(q_dL_dx, th_dL_dx))

        self.assertEqual(sum(r), len(r))