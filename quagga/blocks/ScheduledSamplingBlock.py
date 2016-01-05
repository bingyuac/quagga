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
from quagga.connector import Connector


class ScheduledSamplingBlock(object):
    def __init__(self, probs, true_labels, schedule, seed, device_id=None):
        self.schedule = schedule
        self.rnd = np.random.RandomState(seed)
        self.context = Context(device_id)
        device_id = self.context.device_id

        self.probs = probs.register_usage(device_id)
        self.true_labels = true_labels.register_usage(device_id)
        self.output = Connector(Matrix.empty_like(self.true_labels))

    def fprop(self):
        if self.rnd.binomial(1, self.schedule.value):
            self.output.assign(self.context, self.true_labels)
        else:
            self.probs.argmax(self.context, self.output, axis=1)
        self.output.fprop()