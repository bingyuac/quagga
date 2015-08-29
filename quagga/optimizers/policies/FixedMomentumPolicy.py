import ctypes as ct


class FixedMomentumPolicy(object):
    def __init__(self, momentum):
        self.momentum = ct.c_float(momentum)

    def update(self):
        pass