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
import numpy as np
import ctypes as ct
from itertools import izip
from quagga.matrix import Matrix
from quagga.context import Context


class AdamStep(object):
    def __init__(self, parameters, learning_rate_policy, beta1=0.9, beta2=0.999, epsilon=1e-5):
        self.parameters = parameters
        self.m = []
        self.v = []
        self.contexts = []
        for p in self.parameters:
            m = Matrix.empty_like(p)
            m.sync_fill(0.0)
            self.m.append(m)
            v = Matrix.empty_like(p)
            v.sync_fill(0.0)
            self.v.append(v)
            self.contexts.append(Context(p.device_id))
        self.learning_rate_policy = learning_rate_policy
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.blocking_contexts = []
        self.iteration = 0

    def notify(self):
        self.iteration += 1
        del self.blocking_contexts[:]
        learning_rate = ct.c_float(-self.learning_rate_policy.value)
        learning_rate *= np.sqrt(1 - self.beta2**self.iteration) / (1 - self.beta1**self.iteration)

        for p, m, v, context in izip(self.parameters, self.m, self.v, self.contexts):
            dL_dp = p.backward_matrix
            self.blocking_contexts.append(dL_dp.last_modification_context)
            # m[t+1] = beta1 * m[t] + (1 - beta1) * dL_dp
            m.scale(context, ct.c_float(self.beta1))
            m.add_scaled(context, ct.c_float(1.0 - self.beta1), dL_dp)

            # v[t+1] = beta2 * v[t] + (1 - beta2) * dL_dp^2
            v.add_scaled_hprod(context, dL_dp, dL_dp, self.beta2, (1.0 - self.beta2))

            # p[t+1] = p[t] - learning_rate * m[t+1] / sqrt(v[t+1] + epsilon)
            p.add_scaled_div_sqrt(context, learning_rate, m, v, self.epsilon)