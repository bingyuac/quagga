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


class TrainLossTracker(object):
    def __init__(self, loss_block, period, logger):
        self.loss_block = loss_block
        self.period = period
        self.logger = logger
        self.observers = []
        self.losses = []
        self.iteration = 0
        # we must use this context otherwise we can't guarantee that
        # calculated loss will be correct. Because (very unlikely)
        # probs, true_labels value can be overwritten during calculating loss
        self.context = self.loss_block.context

    def add_observer(self, observer):
        self.observers.append(observer)

    def _accumulate_loss(self):
        loss = self.loss_block.loss
        if type(loss) is list:
            self.losses.extend(loss)
        else:
            self.losses.append(loss)

    def notify(self):
        self.loss_block.calculate_loss(self.context)
        self.context.add_callback(self._accumulate_loss)
        if self.iteration % self.period == 0 and self.iteration != 0:
            self.context.add_callback(self._notify_observers, self.iteration)
        self.iteration += 1

    def _notify_observers(self, iteration):
        loss = np.mean(self.losses)
        self.losses = []
        self.logger.info('Iteration {}: train loss: {:.4f}'.
                         format(iteration, loss))
        for observer in self.observers:
            observer.notify(np.mean(loss))