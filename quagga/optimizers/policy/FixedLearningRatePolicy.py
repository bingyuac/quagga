import ctypes as ct


class FixedLearningRatePolicy(object):
    def __init__(self, learning_rate):
        self.learning_rate = ct.c_float(learning_rate)

    def update(self):
        pass