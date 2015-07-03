from itertools import izip
from quagga.matrix import Matrix


class NagOptimizer(object):
    def __init__(self, max_iter, learning_rate_policy, momentum_policy, model):
        self.max_iter = max_iter
        self.learning_rate_policy = learning_rate_policy
        self.momentum_policy = momentum_policy
        self.model = model
        self.param_u_contexts, self.params = zip(*self.model.params)
        self.grad_o_contexts, self.grads = zip(*self.model.grads)
        self.velocities = []
        for grad in self.grads:
            if type(grad) is not tuple:
                velocity = Matrix.empty_like(grad)
                velocity.fill(0.0)
                self.velocities.append(velocity)
            else:
                self.velocities.append(None)
        self.interruptions = []

    @property
    def learning_rate(self):
        return self.learning_rate_policy.learning_rate

    @property
    def momentum(self):
        return self.momentum_policy.momentum

    def optimize(self):
        for i in xrange(self.max_iter):
            self.model.fprop()
            self.model.bprop()
            self.update()
            self.learning_rate_policy.update()
            self.momentum_policy.update()
            for interruption in self.interruptions:
                if interruption.period % i == 0:
                    interruption.interrupt()

    def update(self):
        for param_u_context, param, grad_o_context, grad, velocity in izip(self.param_u_contexts, self.params, self.grad_o_contexts, self.grads, self.velocities):
            param_u_context.wait(grad_o_context)
            if type(grad) is tuple:
                param.sliced_add_scaled(param_u_context, grad[0], self.learning_rate, grad[1])
            else:
                param.add_scaled(param_u_context, self.learning_rate, grad)

    def add_interruption(self, interruption):
        self.interruptions.append(interruption)