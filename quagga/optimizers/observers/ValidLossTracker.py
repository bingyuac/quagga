import numpy as np


class ValidLossTracker(object):
    def __init__(self, model, period, logger):
        self.model = model
        self.period = period
        self.logger = logger
        self.observers = []
        self.iteration = 0

    def notify(self):
        if self.iteration % self.period == 0 and self.iteration != 0:
            self.model.set_testing_mode()
            losses = []
            try:
                while True:
                    self.model.fprop()
                    losses.append(self.model.loss)
            except StopIteration:
                loss = np.mean(losses)
                self.logger.info('Iteration {}: valid loss: {:.4f}'.
                                 format(self.iteration, loss))
                for observer in self.observers:
                    observer.notify(loss)
            self.model.set_training_mode()
        self.iteration += 1

    def add_observer(self, observer):
        self.observers.append(observer)