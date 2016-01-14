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


class ConditionalValuePolicy(object):
    def __init__(self, initial_value, decay_func, name, logger):
        self.value = initial_value
        self.decay_func = decay_func
        self.name = name
        self.logger = logger
        self.previous_loss = None

    def notify(self, loss):
        if self.previous_loss and loss > self.previous_loss:
            self.value = self.decay_func(self.value)
            self.logger.info('{}: {}'.format(self.name, self.value))
        else:
            self.previous_loss = loss
