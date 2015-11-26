class ConditionalValuePolicy(object):
    def __init__(self, initial_value, decay_func, name, logger):
        self.value = initial_value
        self.decay_func = decay_func
        self.name = name
        self.logger = logger
        self.previous_loss = None

    def notify(self, loss):
        if self.previous_loss and loss > self.previous_loss:
            self.value = self.decay_func(self.value)
            self.logger.info('{}: {}'.format(self.name, self.value))
        else:
            self.previous_loss = loss
