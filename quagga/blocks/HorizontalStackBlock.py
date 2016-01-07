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


class HorizontalStackBlock(object):
    """
    Concatenates input matrices horizontally.

    Parameters
    ----------
    matrices : Matrix (GpuMatrix or CpuMatrix)
        Input matrices that need to be concatenated.
    device_id: int
        Defines the device's id on which the computation will take place
    """

    def __init__(self, *matrices, **kwargs):
        # TODO(sergii): change hsplit to aditive_hsplit for propper gradients accumulation
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
        ncols = [matrix.ncols for matrix in matrices]
        ncols = sum([e for e in ncols[1:]], ncols[0])
        dtype = matrices[0].dtype
        bu_device_id = device_id if self.dL_dmatrices else None
        output = Matrix.empty(matrices[0].nrows, ncols, dtype, device_id)
        self.output = Connector(output, bu_device_id)

    def fprop(self):
        self.output.assign_hstack(self.context, self.matrices)
        self.output.fprop()

    def bprop(self):
        if self.dL_dmatrices:
            col_slices = []
            ncols = [0]
            for matrix, bpropagable in izip(self.matrices, self.bpropagable):
                ncols.append(ncols[-1] + int(matrix.ncols))
                if bpropagable:
                    col_slices.append((ncols[-2], ncols[-1]))
            self.output.backward_matrix.hsplit(self.context, self.dL_dmatrices, col_slices)