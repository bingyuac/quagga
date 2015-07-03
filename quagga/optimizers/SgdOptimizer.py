import ctypes as ct
from itertools import izip


class SgdOptimizer(object):
    def __init__(self, max_iter, learning_rate_policy, model):
        self.max_iter = max_iter
        self.learning_rate_policy = learning_rate_policy
        self.model = model
        self.param_u_contexts, self.params = zip(*self.model.params)
        self.grad_o_contexts, self.grads = zip(*self.model.grads)
        self.interruptions = []
        self.c_dtype = ct.c_float

    @property
    def learning_rate(self):
        learning_rate = self.c_dtype()
        learning_rate.value = -self.learning_rate_policy.learning_rate.value
        return learning_rate

    def optimize(self):
        import time
        total_optimize_time = 0.0
        total_valid_time = 0.0
        t = time.time()
        for i in xrange(self.max_iter):
            self.model.fprop()
            self.model.bprop()
            self.update()
            self.learning_rate_policy.update()
            for interruption in self.interruptions:
                total_optimize_time += time.time() - t
                t = time.time()
                if i % interruption.period == 0:
                    interruption.interrupt(self.model)
                total_valid_time += time.time() - t
                t = time.time()
        print total_optimize_time
        print total_valid_time

    def update(self):
        learning_rate = self.learning_rate
        for param_u_context, param, grad_o_context, grad in izip(self.param_u_contexts, self.params, self.grad_o_contexts, self.grads):
            param_u_context.wait(grad_o_context)
            if type(grad) is tuple:
                param.sliced_add_scaled(param_u_context, grad[0], learning_rate, grad[1])
            else:
                param.add_scaled(param_u_context, learning_rate, grad)

    def add_interruption(self, interruption):
        self.interruptions.append(interruption)