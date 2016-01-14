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
from quagga.context import Context


class SgdStep(object):
    def __init__(self, parameters, learning_rate_policy):
        self.parameters = parameters
        self.learning_rate_policy = learning_rate_policy
        self.contexts = [Context(p.device_id) for p in parameters]
        self.blocking_contexts = []

    def notify(self):
        del self.blocking_contexts[:]
        learning_rate = ct.c_float(-self.learning_rate_policy.value)
        for param, context in izip(self.parameters, self.contexts):
            dL_dparam = param.backward_matrix
            self.blocking_contexts.append(dL_dparam.last_modification_context)
            param.add_scaled(context, learning_rate, dL_dparam)