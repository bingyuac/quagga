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


class AttentionBlock(object):
    """
    Location based attention block
    out = sum_{i=1}^{T}a_i * h_i
    a_i = softmax(h_i * u)
    """
    def __init__(self, matrices, u, mask=None, device_id=None):
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.output = Matrix.empty_like(matrices[0], device_id)
        learning = matrices[0].bpropagable or u.bpropagable
        self.output = Connector(self.output, device_id if learning else None)
        if matrices[0].bpropagable:
            self.matrices, self.dL_dmatrices = \
                izip(*matrices.register_usage(device_id, device_id))
        else:
            self.matrices = matrices.register_usage(device_id)
        self.length = matrices.length
        if u.bpropagable:
            self.u, self.dL_du = u.register_usage(device_id, device_id)
        else:
            self.u = u.register_usage(device_id)
        if mask:
            self.mask = mask.register_usage(device_id)
        self.a = Matrix.empty(matrices[0].nrows, matrices.length,
                              'float', device_id)
        self.dL_dpre_a = Matrix.empty_like(self.a)
        self.a_cols = [self.a[:, i] for i in xrange(len(self.matrices))]

    def fprop(self):
        for i in xrange(self.length):
            self.a_cols[i].assign_dot(self.context, self.matrices[i], self.u)
        if hasattr(self, 'mask'):
            self.a.fill(self.context, -3.402823466e+38, self.mask, 0.0)
        self.a.softmax(self.context, self.a)
        self.output.assign_sequential_weighted_sum(self.context, self.a,
                                                   self.matrices[:self.length])
        self.output.fprop()

    def bprop(self):
        dL_doutput = self.output.backward_matrix
        self.dL_dpre_a.assign_dL_dpre_a(self.context, dL_doutput, self.a,
                                        self.matrices[:self.length])
        if hasattr(self, 'dL_dmatrices'):
            Matrix.add_attention_tile(self.context, dL_doutput, self.a,
                                      self.dL_dpre_a, self.u,
                                      self.dL_dmatrices[:self.length])
        if hasattr(self, 'dL_du'):
            self.dL_du.add_attention_derivative(self.context, self.dL_dpre_a,
                                                self.matrices[:self.length])