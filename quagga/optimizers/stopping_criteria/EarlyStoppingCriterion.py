class EarlyStoppingCriterion(object):
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.iteration = 0

    def is_fulfilled(self):
        return True if self.iteration > self.max_iter else False

    def notify(self):
        self.iteration += 1