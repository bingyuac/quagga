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


class DropoutBlock(object):
    """
    Sets elements of input matrix ``x`` to zero with probability
    ``dropout_prob`` in training mode. Scales ``x`` by factor of
    ``1-dropout_prob`` during in testing mode.

    Parameters
    ----------
    dropout_prob : float
    x : :class:`~quagga.matrix.CpuMatrix` or :class:`~quagga.matrix.GpuMatrix`
    seed : int
    device_id : int
        Defines the device's id on which the computation will take place

    Notes
    -----
    The dropout block is a regularizer that randomly sets input values to zero
    in training mode. This procedure is supposed to improve generalization.
    During testing, the dropout block scales input values.
    """
    def __init__(self, dropout_prob, x, seed=42, device_id=None):
        self.dropout_prob = dropout_prob
        self.f_context = Context(device_id)
        device_id = self.f_context.device_id
        self.generator = Matrix.get_random_generator(seed)
        if x.bpropagable:
            self.b_context = Context(device_id)
            self.x, self.dL_dx = x.register_usage(device_id, device_id)
        else:
            self.x = x.register_usage(device_id)
        self.output = Matrix.empty_like(self.x)
        self.output = Connector(self.output, device_id if x.bpropagable else None)
        self.training_mode = True

    def fprop(self):
        if self.training_mode:
            self.x.dropout(self.f_context, self.generator, self.dropout_prob, self.output)
        else:
            self.x.scale(self.f_context, 1.0 - self.dropout_prob, self.output)
        self.output.fprop()

    def bprop(self):
        if hasattr(self, 'dL_dx') and self.training_mode:
            dL_doutput = self.output.backward_matrix
            self.dL_dx.add_mask_zeros(self.b_context, dL_doutput, self.output)

    def set_training_mode(self):
        self.training_mode = True

    def set_testing_mode(self):
        self.training_mode = False