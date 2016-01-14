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


class ValidLossTracker(object):
    def __init__(self, loss_block, logger):
        self.loss_block = loss_block
        self.logger = logger
        self.observers = []
        self.losses = []
        # we must use this context otherwise we can't guarantee that
        # calculated loss will be correct
        self.context = self.loss_block.context

    def add_observer(self, observer):
        self.observers.append(observer)

    def accumulate_loss(self):
        loss = self.loss_block.loss
        if type(loss) is list:
            self.losses.extend(loss)
        else:
            self.losses.append(loss)

    def notify_about_fprop(self):
        self.loss_block.calculate_loss(self.context)
        self.context.add_callback(self.accumulate_loss)

    def notify(self, iteration):
        self.context.add_callback(self._notify, iteration)

    def _notify(self, iteration):
        loss = np.mean(self.losses)
        self.losses = []
        self.logger.info('Iteration {}: valid loss: {:.4f}'.
                         format(iteration, loss))
        for observer in self.observers:
            observer.notify(loss)