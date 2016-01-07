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
import ctypes as ct
from quagga.context import Context


class GradientReversalBlock(object):
    def __init__(self, x, scale_factor=1.0):
        self.context = Context(x.device_id)
        device_id = self.context.device_id
        self.output = x
        if x.bpropagable:
            _, self.dL_dx = x.register_usage(device_id, device_id)
        self.scale_factor = ct.c_float(-scale_factor)

    def bprop(self):
        self.dL_dx.scale(self.context, self.scale_factor)