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


class LastSelectorBlock(object):
    def __init__(self, x):
        device_id = x[0].device_id
        learning = x[0].bpropagable
        self.context = Context(device_id)
        self.output = Matrix.empty_like(x[0])
        self.output = Connector(self.output, device_id if learning else None)
        if learning:
            self.x, self.dL_dx = izip(*x.register_usage(device_id, device_id))
        else:
            self.x = x.register_usage(device_id)
        self.last_idx = x.length - 1

    def fprop(self):
        self.output.assign(self.context, self.x[self.last_idx])
        self.output.fprop()

    def bprop(self):
        self.dL_dx[self.last_idx].add(self.context, self.output.backward_matrix)