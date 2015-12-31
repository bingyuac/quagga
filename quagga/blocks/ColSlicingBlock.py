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
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class ColSlicingBlock(object):
    """
    Parameters
    ----------
    W
    col_indexes

    Returns
    -------
    """
    def __init__(self, W, col_indexes):
        device_id = W.device_id
        self.context = Context(device_id)
        learning = W.bpropagable
        if learning:
            self.W, self.dL_dW = W.register_usage_with_sparse_backward_matrix()
        else:
            self.W = W.register_usage(device_id)
        self.col_indexes = col_indexes.register_usage(device_id)
        output = Matrix.empty(W.nrows, col_indexes.ncols, device_id=device_id)
        self.output = Connector(output, device_id if learning else None)

    def fprop(self):
        self.W.slice_columns(self.context, self.col_indexes, self.output)
        self.output.fprop()

    def bprop(self):
        if hasattr(self, 'dL_dW'):
            self.dL_dW.add_columns_slice(self.col_indexes, self.output.bprop())