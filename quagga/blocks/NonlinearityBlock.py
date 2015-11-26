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


class NonlinearityBlock(object):
    def __init__(self, x, nonlinearity, device_id=None):
        self.f_context = Context(device_id)
        device_id = self.f_context.device_id
        self.learning = x.bpropagable
        if self.learning:
            self.b_context = Context(device_id)
            self.x, self.dL_dx = x.register_usage(device_id, device_id)
            self._df_dpref = Matrix.empty_like(self.x, device_id)
        else:
            self.x = x.register_usage(device_id)
        output = Matrix.empty_like(x, device_id)
        self.output = Connector(output, device_id if self.learning else None)
        if nonlinearity == 'sigmoid':
            self.f = self.x.sigmoid
        elif nonlinearity == 'tanh':
            self.f = self.x.tanh
        elif nonlinearity == 'relu':
            self.f = self.x.relu
        else:
            raise ValueError('TODO!')
        self.training_mode = True

    @property
    def df_dpref(self):
        if self.training_mode and self.learning:
            return self._df_dpref

    def fprop(self):
        self.f(self.f_context, self.output, self.df_dpref)
        self.output.fprop()

    def bprop(self):
        if hasattr(self, 'dL_dx'):
            # dL/dpref = dL/df .* df/dpref
            dL_df = self.output.backward_matrix
            self.dL_dx.add_hprod(self.b_context, dL_df, self.df_dpref)

    def set_training_mode(self):
        self.training_mode = True

    def set_testing_mode(self):
        self.training_mode = False