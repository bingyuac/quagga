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


class ArgmaxBlock(object):
    """
    Determines argmax values along the specified axis in the input matrix. The
    block returns a vector (matrix with one of its dimensions equals 1) of argmax
    values.


    Parameters
    ----------
    x : Matrix (GpuMatrix or CpuMatrix)
        Block's input
    axis : int
        Axis along which argmax is determined
    device_id : int
        Defines the device's id on which the computation will take place

    Returns
    -------
    vector
        A vector containing argmax values (e.g. argmax for each row if axis == 1).
    """
    def __init__(self, x, axis, device_id=None):
        if axis != 1:
            raise NotImplementedError
        self.axis = axis
        self.context = Context(device_id)
        device_id = self.context.device_id

        self.x = x.register_usage(device_id)
        self.output = Connector(Matrix.empty(x.nrows, 1, x.dtype, device_id))

    def fprop(self):
        self.x.argmax(self.context, self.output, self.axis)
        self.output.fprop()