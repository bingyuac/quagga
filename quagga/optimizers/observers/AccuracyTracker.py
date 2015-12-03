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


class AccuracyTracker(object):
    def __init__(self, model, period, logger):
        self.model = model
        self.period = period
        self.logger = logger
        self.observers = []
        self.iteration = 0
        self.calculate_accuracy = Context.callback(self.calculate_accuracy)
        self.log_notify = Context.callback(self.log_notify)

    def notify(self):
        if self.iteration % self.period == 0 and self.iteration != 0:
            self.model.set_testing_mode()
            accuracy = []
            # we must use this context otherwise we can't guarantee that
            # calculated loss will be correct
            context = self.model.loss_block.context
            loss_block = self.model.loss_block
            try:
                while True:
                    self.model.fprop()
                    probs = loss_block.probs.to_host(context)
                    true_labels = loss_block.true_labels.to_host(context)
                    context.add_callback(self.calculate_accuracy, accuracy, probs, true_labels)
            except StopIteration:
                context.add_callback(self.log_notify, accuracy, self.iteration)
            self.model.set_training_mode()
        self.iteration += 1

    def add_observer(self, observer):
        self.observers.append(observer)

    def calculate_accuracy(self, accuracy, probs, true_labels):
        if true_labels.shape[1] == 1:
            predicted_idx = np.argmax(probs, axis=1)
            true_labels = true_labels[0]
            accuracy.append(np.sum(predicted_idx == true_labels) / float(len(true_labels)))
        else:
            # TODO(sergii)
            pass

    def log_notify(self, accuracy, iteration):
        accuracy = np.mean(accuracy)
        self.logger.info('Iteration {}: valid accuracy: {:.4f}'.
                         format(iteration, accuracy))
        for observer in self.observers:
            observer.notify(accuracy)