class FixedLearningRatePolicy(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def notify(self):
        pass