class ScheduledValuePolicy(object):
    def __init__(self, schedule, logger, name):
        self.schedule = schedule
        self.logger = logger
        self.name = name
        self.iteration = 0
        self.value = None

    def notify(self):
        if self.iteration in self.schedule:
            self.value = self.schedule[self.iteration]
            self.logger.info('{}: {}'.format(self.name, self.value))
        self.iteration += 1