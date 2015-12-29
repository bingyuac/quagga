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
from quagga.matrix import Matrix
from quagga.context import Context


class SigmoidBlock(object):
    """
    Sigmoid nonlinearity
    """

    def __init__(self, x, device_id=None):
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.x = x.register_usage(device_id)
        self.probs = Matrix.empty_like(self.x)

    def fprop(self):
        self.x.sigmoid(self.context, self.probs)