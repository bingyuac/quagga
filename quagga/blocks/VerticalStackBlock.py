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


class VerticalStackBlock(object):
    """
    Concatenates matrices vertically.
    """

    def __init__(self, *matrices, **kwargs):
        # TODO(sergii): change vsplit to aditive_vsplit for propper gradients accumulation
        self.context = Context(kwargs.get('device_id'))
        device_id = self.context.device_id
        self.matrices = []
        self.dL_dmatrices = []
        self.bpropagable = []
        for matrix in matrices:
            self.bpropagable.append(matrix.bpropagable)
            if matrix.bpropagable:
                matrix, dL_dmatrix = matrix.register_usage(device_id, device_id)
                self.dL_dmatrices.append(dL_dmatrix)
            else:
                matrix = matrix.register_usage(device_id)
            self.matrices.append(matrix)
        nrows = sum(matrix.nrows for matrix in matrices)
        dtype = matrices[0].dtype
        bu_device_id = device_id if self.dL_dmatrices else None
        output = Matrix.empty(nrows, matrices[0].ncols, dtype, device_id)
        self.output = Connector(output, bu_device_id)

    def fprop(self):
        self.output.assign_vstack(self.context, self.matrices)
        self.output.fprop()

    def bprop(self):
        if self.dL_dmatrices:
            row_slices = []
            nrows = [0]
            for matrix, bpropagable in izip(self.matrices, self.bpropagable):
                nrows.append(nrows[-1] + int(matrix.nrows))
                if bpropagable:
                    row_slices.append((nrows[-2], nrows[-1]))
            self.output.backward_matrix.vsplit(self.context, self.dL_dmatrices, row_slices)