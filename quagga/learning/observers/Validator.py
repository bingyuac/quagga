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


class Validator(object):
    def __init__(self, model, period):
        self.model = model
        self.period = period
        self.observers = []
        self.fprop_ovservers = []
        self.iteration = 0

    def add_observer(self, observer):
        self.observers.append(observer)

    def add_fprop_observer(self, observer):
        self.fprop_ovservers.append(observer)

    def notify(self):
        if self.iteration % self.period == 0 and self.iteration != 0:
            self.model.set_testing_mode()
            try:
                while True:
                    self.model.fprop()
                    for observer in self.fprop_ovservers:
                        observer.notify_about_fprop()
            except StopIteration:
                for observer in self.observers:
                    observer.notify(self.iteration)
            self.model.set_training_mode()
        self.iteration += 1