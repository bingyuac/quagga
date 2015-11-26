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
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class SequentialSumPoolingBlock(object):
    # TODO(sergii): change sequentially_tile to add_sequentially_tile, because can erase gradients
    def __init__(self, matrices, device_id=None):
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.output = Matrix.empty_like(matrices[0], device_id)
        learning = matrices[0].bpropagable
        self.output = Connector(self.output, device_id if learning else None)
        if learning:
            self.matrices, self.dL_dmatrices = izip(*matrices.register_usage(device_id, device_id))
        else:
            self.matrices = matrices.register_usage(device_id)
        self.length = matrices.length

    def fprop(self):
        self.output.assign_sequential_sum_pooling(self.context, self.matrices[:self.length])
        self.output.fprop()

    def bprop(self):
        dL_doutput = self.output.backward_matrix
        Matrix.sequentially_tile(self.context, dL_doutput, self.dL_dmatrices[:self.length])