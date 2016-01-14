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
from itertools import izip
from quagga.matrix import Matrix
from quagga.context import Context


class MomentumStep(object):
    def __init__(self, parameters, learning_rate_policy, momentum_policy):
        self.parameters = parameters
        self.velocity = []
        for p in self.parameters:
            v = Matrix.empty_like(p)
            v.sync_fill(0.0)
            self.velocity.append(v)
        self.learning_rate_policy = learning_rate_policy
        self.momentum_policy = momentum_policy
        self.contexts = [Context(p.device_id) for p in parameters]
        self.blocking_contexts = []

    def notify(self):
        del self.blocking_contexts[:]
        learning_rate = ct.c_float(-self.learning_rate_policy.value)
        momentum = ct.c_float(self.momentum_policy.value)
        for p, v, context in izip(self.parameters, self.velocity, self.contexts):
            dL_dp = p.backward_matrix
            self.blocking_contexts.append(dL_dp.last_modification_context)
            v.scale(context, momentum)
            v.add_scaled(context, learning_rate, dL_dp)
            p.add(context, v)