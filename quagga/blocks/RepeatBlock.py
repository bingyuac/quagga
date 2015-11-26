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


class RepeatBlock(object):
    def __init__(self, x, repeats, axis=None, device_id=None):
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.repeats = repeats
        self.axis = axis
        learning = x.bpropagable
        if learning:
            self.x, self.dL_dx = x.register_usage(device_id, device_id)
        else:
            self.x = x.register_usage(device_id)
        if axis == 0:
            self.output = Matrix.empty(x.nrows * repeats, x.ncols, x.dtype, device_id)
        elif axis == 1:
            self.output = Matrix.empty(x.nrows, x.ncols * repeats, x.dtype, device_id)
        else:
            raise ValueError('TODO')
        self.output = Connector(self.output, device_id if learning else None)

    def fprop(self):
        self.output.assign_repeat(self.context, self.x, self.repeats, self.axis)
        self.output.fprop()

    def bprop(self):
        if hasattr(self, 'dL_dx'):
            self.dL_dx.add_repeat_derivative(self.context, self.output.backward_matrix, self.repeats, self.axis)