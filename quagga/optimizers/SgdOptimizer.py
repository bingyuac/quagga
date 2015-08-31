import ctypes as ct
from itertools import izip


class SgdOptimizer(object):
    def __init__(self, stopping_criterion, learning_rate_policy, model):
        self.stopping_criterion = stopping_criterion
        self.learning_rate_policy = learning_rate_policy
        self.model = model
        self.param_u_contexts, self.params = zip(*self.model.params)
        self.grad_o_contexts, self.grads = zip(*self.model.grads)
        self.oververs = []

    def optimize(self):
        self.model.set_training_mode()
        while not self.stopping_criterion.is_fulfilled():
            self.model.fprop()
            self.model.bprop()
            self.update()
            self.learning_rate_policy.update()
            for observer in self.oververs:
                observer.notify()

    def update(self):
        learning_rate = ct.c_float(-self.learning_rate_policy.learning_rate)
        for param_u_context, param, grad_o_context, grad in izip(self.param_u_contexts, self.params, self.grad_o_contexts, self.grads):
            param_u_context.wait(grad_o_context)
            if isinstance(grad, tuple):
                if isinstance(grad[1], list):
                    param.sliced_rows_batch_scaled_add(param_u_context, grad[0], learning_rate, grad[1])
                else:
                    raise ValueError('TODO!')
            else:
                param.add_scaled(param_u_context, learning_rate, grad)

    def add_observer(self, observer):
        self.oververs.append(observer)