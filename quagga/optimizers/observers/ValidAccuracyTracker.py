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


class ValidAccuracyTracker(object):
    def __init__(self, logger):
        self.observers = []
        self.logger = logger
        self.accuracy = []

    def calculate(self, context, loss_block):
        probs = loss_block.probs.to_host(context)
        true_labels = loss_block.true_labels.to_host(context)
        context.add_callback(self.calculate_accuracy, probs, true_labels)

    def calculate_accuracy(self, probs, true_labels):
        if true_labels.shape[1] == 1:
            if probs.shape[1] == 1:
                # sigmoid
                predicted_idx = probs[:, 0] > 0.5
            else:
                # softmax, true_labels integer vector
                predicted_idx = np.argmax(probs, axis=1)
            true_labels = true_labels[:, 0]
            self.accuracy.append(np.sum(predicted_idx == true_labels) /
                                 float(len(true_labels)))
        else:
            # TODO(sergii)
            pass

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify(self, iteration):
        accuracy = np.mean(self.accuracy)
        self.accuracy = []
        self.logger.info('Iteration {}: valid accuracy: {:.4f}'.
                         format(iteration, accuracy))
        for observer in self.observers:
            observer.notify(accuracy)
