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
class Optimizer(object):
    def __init__(self, stopping_criterion, model):
        self.stopping_criterion = stopping_criterion
        self.model = model
        self.observers = []

    def optimize(self):
        self.model.set_training_mode()
        while not self.stopping_criterion.is_fulfilled():
            self.model.fprop()
            self.model.bprop()
            for observer in self.observers:
                observer.notify()

    def add_observer(self, observer):
        self.observers.append(observer)