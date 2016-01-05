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


class L2RegularizationBlock(object):
    """
    Forms a regularization term for `x` with `regularization_value` as a
    lambda term.

    Parameters
    ----------
    x : Matrix (GpuMatrix or CpuMatrix)
        Input matrix
    regularization_value : float
        Lambda term

    Notes
    -----
    This reguralization is used during backpropagation only.

    """
    def __init__(self, x, regularization_value):
        self.context = Context(x.device_id)
        device_id = self.context.device_id
        if x.bpropagable:
            self.x, self.dL_dx = x.register_usage(device_id, device_id)
        else:
            self.x = x.register_usage(device_id)
        self.reg_value = ct.c_float(2 * regularization_value)

    def bprop(self):
        self.dL_dx.add_scaled(self.context, self.reg_value, self.x)