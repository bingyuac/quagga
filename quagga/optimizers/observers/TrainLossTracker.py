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
        self.losses = []
        self.iteration = 0
        self.accumulate_loss = Context.callback(self.accumulate_loss)
        self.log_notify = Context.callback(self.log_notify)

    def notify(self):
        self.loss_block.calculate_loss(self.context)
        self.context.add_callback(self.accumulate_loss)
        if self.iteration % self.period == 0 and self.iteration != 0:
            self.context.add_callback(self.log_notify, self.iteration)
        self.iteration += 1

    def accumulate_loss(self):
        loss = self.loss_block.loss
        if type(loss) is list:
            self.losses.extend(loss)
        else:
            self.losses.append(loss)

    def log_notify(self, iteration):
        self.logger.info('Iteration {}: train loss: {:.4f}'.
                         format(iteration, np.mean(self.losses)))
        self.losses = []