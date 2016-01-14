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
from itertools import chain

from quagga.utils import List
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class SequentialHorizontalStackBlock(object):
    def __init__(self, x_sequence, y_sequence, device_id=None):
        """
        TODO
        """
        # TODO add during hsplit otherwise wrong accumulation of gradients
        if all(e.bpropagable for e in chain(x_sequence, y_sequence)):
            learning = True
        elif all(not e.bpropagable for e in chain(x_sequence, y_sequence)):
            learning = False
        else:
            raise ValueError('All elements should be bpropagable or '
                             'non-bpropagable. Mixed state is not allowed!')
        x_ncols = x_sequence[0].ncols
        y_ncols = y_sequence[0].ncols
        dtype = x_sequence[0].dtype
        for x, y in izip(x_sequence, y_sequence):
            if x.ncols != x_ncols or y.ncols != y_ncols:
                raise ValueError("All matrices in the sequence should have the same number of columns!")
            if x.nrows != y.nrows:
                raise ValueError("Can't stack matrices in sequence with different number of rows!")
            if x.dtype != dtype or y.dtype != dtype:
                raise ValueError("Can't stack matrices with different dtypes!")

        self.context = Context(device_id)
        device_id = self.context.device_id
        if learning:
            self.x_sequence, self.dL_dx_sequences = izip(*x_sequence.register_usage(device_id, device_id))
            self.y_sequence, self.dL_dy_sequences = izip(*y_sequence.register_usage(device_id, device_id))
            self.dL_dx_sequences = List(self.dL_dx_sequences, x_sequence.length)
            self.dL_dy_sequences = List(self.dL_dy_sequences, y_sequence.length)
        else:
            self.x_sequence = x_sequence.register_usage(device_id)
            self.y_sequence = y_sequence.register_usage(device_id)
        self.x_sequence = List(self.x_sequence, x_sequence.length)
        self.y_sequence = List(self.y_sequence, y_sequence.length)
        output = []
        for _ in xrange(x_sequence.length):
            matrix = Matrix.empty(x_sequence[0].nrows, x_ncols + y_ncols, dtype, device_id)
            output.append(Connector(matrix, device_id))
        self.output = List(output, x_sequence.length)
        if learning:
            self.dL_dx_sequences = List(self.dL_dx_sequences, x_sequence.length)
            self.dL_dy_sequences = List(self.dL_dy_sequences, x_sequence.length)

    def fprop(self):
        Matrix.batch_hstack(self.context, self.x_sequence, self.y_sequence, self.output)

    def bprop(self):
        dL_doutput_sequence = [e.backward_matrix for e in self.output]
        Matrix.batch_hsplit(self.context, dL_doutput_sequence, self.dL_dx_sequences, self.dL_dy_sequences)