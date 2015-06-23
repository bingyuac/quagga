from itertools import izip
from quagga.context import Context
from quagga.connector import Connector


class SgdOptimizer(object):
    def __init__(self, learning_rate, model):
        self.model = model
        self.param_u_contexts, self.params = zip(*self.model.params)
        self.grad_o_contexts, self.grads = zip(*self.model.grads)
        self.learning_rates = []
        c = Context(learning_rate.device_id)
        learning_rate.negate(c)
        learning_rate = Connector(learning_rate, c)
        for u_context in self.param_u_contexts:
            self.learning_rates.append(learning_rate.register_usage(u_context))
        learning_rate.fprop()

    def optimize(self):
        for i in xrange(1000):
            self.model.fprop()
            self.model.bprop()
            self.update()
        self.model.fprop()

    def update(self):
        for learning_rate, param_u_context, param, grad_o_context, grad in izip(self.learning_rates, self.param_u_contexts, self.params, self.grad_o_contexts, self.grads):
            param_u_context.wait(grad_o_context)
            if type(grad) is tuple:
                param.sliced_add_scaled(param_u_context, grad[0], learning_rate, grad[1])
            else:
                param.add_scaled(param_u_context, learning_rate, grad)