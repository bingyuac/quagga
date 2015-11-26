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


class ValidLossTracker(object):
    def __init__(self, model, period, logger):
        self.model = model
        self.period = period
        self.logger = logger
        self.observers = []
        self.iteration = 0
        self.accumulate_loss = Context.callback(self.accumulate_loss)
        self.log_notify = Context.callback(self.log_notify)

    def notify(self):
        if self.iteration % self.period == 0 and self.iteration != 0:
            self.model.set_testing_mode()
            losses = []
            # we must use this context otherwise we can't guarantee that
            # calculated loss will be correct
            context = self.model.loss_block.context
            loss_block = self.model.loss_block
            try:
                while True:
                    self.model.fprop()
                    loss_block.calculate_loss(context)
                    context.add_callback(self.accumulate_loss, losses)
            except StopIteration:
                context.add_callback(self.log_notify, losses, self.iteration)
            self.model.set_training_mode()
        self.iteration += 1

    def add_observer(self, observer):
        self.observers.append(observer)

    def accumulate_loss(self, losses):
        loss = self.model.loss_block.loss
        if type(loss) is list:
            losses.extend(loss)
        else:
            losses.append(loss)

    def log_notify(self, losses, iteration):
        loss = np.mean(losses)
        self.logger.info('Iteration {}: valid loss: {:.4f}'.
                         format(iteration, loss))
        for observer in self.observers:
            observer.notify(loss)