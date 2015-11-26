class ScheduledMomentumPolicy(object):
    def __init__(self, schedule, logger):
        self.schedule = schedule
        self.logger = logger
        self.iteration = 0
        self.momentum = None

    def notify(self):
        if self.iteration in self.schedule:
            self.momentum = self.schedule[self.iteration]
            self.logger.info('momentum: {}'.format(self.momentum))
        self.iteration += 1