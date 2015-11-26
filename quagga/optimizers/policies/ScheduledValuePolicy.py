class ScheduledValuePolicy(object):
    def __init__(self, schedule, name, logger):
        self.schedule = schedule
        self.name = name
        self.logger = logger
        self.iteration = 0
        self.value = None

    def notify(self):
        if self.iteration in self.schedule:
            self.value = self.schedule[self.iteration]
            self.logger.info('{}: {}'.format(self.name, self.value))
        self.iteration += 1