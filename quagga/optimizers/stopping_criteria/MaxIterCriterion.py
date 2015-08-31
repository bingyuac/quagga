class MaxIterCriterion(object):
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.iteration = 0

    def is_fulfilled(self):
        self.iteration += 1
        return True if self.iteration > self.max_iter else False