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
from quagga.context import Context


class TrainLossTracker(object):
    def __init__(self, model, period, logger):
        self.loss_block = model.loss_block
        # we must use this context otherwise we can't guarantee that
        # calculated loss will be correct. Because (very unlikely)
        # probs, true_labels value can be overwritten during calculating loss
        self.context = self.loss_block.context
        self.period = period
        self.logger = logger
        self.observers = []
        self.losses = []
        self.iteration = 0

    def notify(self):
        self.loss_block.calculate_loss(self.context)
        self.context.add_callback(self.accumulate_loss)
        if self.iteration % self.period == 0 and self.iteration != 0:
            self.context.add_callback(self.log_notify, self.iteration)
        self.iteration += 1

    def add_observer(self, observer):
        self.observers.append(observer)

    def accumulate_loss(self):
        loss = self.loss_block.loss
        if type(loss) is list:
            self.losses.extend(loss)
        else:
            self.losses.append(loss)

    def log_notify(self, iteration):
        loss = np.mean(self.losses)
        self.losses = []
        self.logger.info('Iteration {}: train loss: {:.4f}'.
                         format(iteration, loss))
        for observer in self.observers:
            observer.notify(np.mean(loss))