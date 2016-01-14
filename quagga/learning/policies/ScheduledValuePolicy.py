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
class ScheduledValuePolicy(object):
    def __init__(self, schedule, name, logger):
        self.schedule = schedule
        self.name = name
        self.logger = logger
        self.iteration = 0
        self.value = None

    def notify(self):
        if self.iteration in self.schedule:
            self.value = self.schedule[self.iteration]
            self.logger.info('{}: {}'.format(self.name, self.value))
        self.iteration += 1