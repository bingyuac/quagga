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
import numpy as np
from itertools import izip
from unittest import TestCase
from quagga.matrix import Matrix
from quagga.connector import Connector
from quagga.blocks import SoftmaxCeBlock
from quagga.blocks import VerticalStackBlock


class TestVerticalStackBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 15

    def test_fprop(self):
        r = []
        for i in xrange(self.N):
            matrices = []
            ncols = self.rng.random_integers(1, 5000)
            for _ in xrange(self.rng.random_integers(1, 10)):
                nrows = self.rng.random_integers(1, 5000)
                matrices.append(self.rng.rand(nrows, ncols).astype(np.float32))

            output = {}
            for processor_type in ['gpu', 'cpu']:
                quagga.processor_type = processor_type
                qmatrices = [Connector(Matrix.from_npa(m)) for m in matrices]
                for m in qmatrices:
                    m.fprop()
                vstack_block = VerticalStackBlock(*qmatrices)
                vstack_block.fprop()
                output[processor_type] = vstack_block.output.to_host()

            r.append(np.allclose(output['gpu'], output['cpu']))

        self.assertEqual(sum(r), self.N)

    def test_numpy_fprop(self):
        r = []
        quagga.processor_type = 'gpu'
        for _ in xrange(self.N):
            matrices = []
            ncols = self.rng.random_integers(1, 5000)
            for _ in xrange(self.rng.random_integers(1, 10)):
                nrows = self.rng.random_integers(1, 5000)
                matrices.append(self.rng.rand(nrows, ncols).astype(np.float32))

            numpy_output = np.vstack([m for m in matrices])
            matrices = [Connector(Matrix.from_npa(m)) for m in matrices]
            vstack_block = VerticalStackBlock(*matrices)
            for m in matrices:
                m.fprop()
            vstack_block.fprop()
            quagga_output = vstack_block.output.to_host()

            r.append(np.allclose(numpy_output, quagga_output))
        self.assertEqual(sum(r), self.N)

    def test_bprop(self):
        r = []
        for i in xrange(self.N):
            matrices = []
            ncols = self.rng.random_integers(1, 3000)
            nrows = [0]
            row_slices = []
            device_ids = []
            for _ in xrange(self.rng.random_integers(1, 10)):
                _nrows = self.rng.random_integers(1, 2000)
                nrows.append(nrows[-1] + _nrows)
                if self.rng.choice([True, False]):
                    device_ids.append(0)
                    row_slices.append((nrows[-2], nrows[-1]))
                else:
                    device_ids.append(None)
                matrices.append(self.rng.rand(_nrows, ncols).astype(np.float32))
            true_labels = self.rng.randint(ncols, size=(nrows[-1], 1)).astype(np.int32)
            if not row_slices:
                r.append(True)
                continue

            output = {}
            for processor_type in ['gpu', 'cpu']:
                quagga.processor_type = processor_type
                qmatrices = [Connector(Matrix.from_npa(m), d_id) for m, d_id in izip(matrices, device_ids)]
                qtrue_labels = Connector(Matrix.from_npa(true_labels))
                vstack_block = VerticalStackBlock(*qmatrices)
                sce_block = SoftmaxCeBlock(vstack_block.output, qtrue_labels)

                for m in qmatrices:
                    m.fprop()
                qtrue_labels.fprop()
                vstack_block.fprop()
                sce_block.fprop()
                sce_block.bprop()
                vstack_block.bprop()

                output[processor_type] = [m.backward_matrix.to_host()
                                          for m in qmatrices if m.bpropagable]

            for dL_dm_gpu, dL_dm_cpu in izip(output['gpu'], output['cpu']):
                if not np.allclose(dL_dm_gpu, dL_dm_cpu):
                    r.append(False)
                    break
            else:
                r.append(True)
        self.assertEqual(sum(r), self.N)
