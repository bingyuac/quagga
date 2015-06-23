from itertools import izip


class SgdOptimizer(object):
    def __init__(self, learning_rate, model):
        self.model = model
        self.param_u_contexts, self.params = zip(*self.model.params)
        self.grad_o_contexts, self.grads = zip(*self.model.grads)
        learning_rate.value = -learning_rate.value
        self.learning_rate = learning_rate

    def optimize(self):
        for i in xrange(1000):
            self.model.fprop()
            self.model.bprop()
            self.update()
        self.model.fprop()

    def update(self):
        for param_u_context, param, grad_o_context, grad in izip(self.param_u_contexts, self.params, self.grad_o_contexts, self.grads):
            param_u_context.wait(grad_o_context)
            if type(grad) is tuple:
                param.sliced_add_scaled(param_u_context, grad[0], self.learning_rate, grad[1])
            else:
                param.add_scaled(param_u_context, self.learning_rate, grad)