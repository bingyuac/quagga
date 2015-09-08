import numpy as np


class TrainLossTracker(object):
    def __init__(self, model, period, logger):
        self.model = model
        self.period = period
        self.logger = logger
        self.losses = []
        self.iteration = 0

    def notify(self):
        loss = self.model.loss
        if type(loss) is list:
            self.losses.extend(loss)
        else:
            self.losses.append(loss)
        if self.iteration % self.period == 0 and self.iteration != 0:
            self.logger.info('Iteration {}: train loss: {:.4f}'.
                             format(self.iteration, np.mean(self.losses)))
            self.losses = []
        self.iteration += 1