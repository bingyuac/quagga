import ctypes as ct
from itertools import izip
from quagga.context import Context


class SparseSgdStep(object):
    def __init__(self, parameters, learning_rate_policy):
        self.parameters = parameters
        self.learning_rate_policy = learning_rate_policy
        self.contexts = [Context(p.device_id) for p in parameters]
        self.blocking_contexts = []

    def notify(self):
        del self.blocking_contexts[:]
        learning_rate = ct.c_float(-self.learning_rate_policy.learning_rate)
        for param, context in izip(self.parameters, self.contexts):
            dL_dparam = param.backward_matrix
            self.blocking_contexts.extend(dL_dparam.get_last_modification_contexts())
            param.add_scaled(context, learning_rate, dL_dparam)