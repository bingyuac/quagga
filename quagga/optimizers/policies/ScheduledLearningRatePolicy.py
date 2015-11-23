class ScheduledLearningRatePolicy(object):
    def __init__(self, schedule):
        self.schedule = schedule
        self.iteration = 0
        self.learning_rate = None

    def notify(self):
        if self.iteration in self.schedule:
            self.learning_rate = self.schedule[self.iteration]
        self.iteration += 1