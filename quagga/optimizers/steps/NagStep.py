import ctypes as ct
from itertools import izip
from quagga.matrix import Matrix
from quagga.context import Context


# https://github.com/lisa-lab/pylearn2/pull/136
class NagStep(object):
    def __init__(self, parameters, learning_rate_policy, momentum_policy):
        self.parameters = parameters
        self.velocity = []
        for p in self.parameters:
            v = Matrix.empty_like(p)
            v.sync_fill(0.0)
            self.velocity.append(v)
        self.learning_rate_policy = learning_rate_policy
        self.momentum_policy = momentum_policy
        self.contexts = [Context(p.device_id) for p in parameters]
        self.blocking_contexts = []

    def notify(self):
        del self.blocking_contexts[:]
        learning_rate = ct.c_float(-self.learning_rate_policy.learning_rate)
        momentum = ct.c_float(self.momentum_policy.momentum)
        for p, v, context in izip(self.parameters, self.velocity, self.contexts):
            dL_dp = p.backward_matrix
            self.blocking_contexts.append(dL_dp.last_modification_context)
            # v_t+1 = momentum * v_t - learning_rate * dL_dp
            v.scale(context, momentum)
            v.add_scaled(context, learning_rate, dL_dp)
            # p_t+1 = momentum * v_t+1 - learning_rate * dL_dp
            p.add_scaled(context, momentum, v)
            p.add_scaled(context, learning_rate, dL_dp)