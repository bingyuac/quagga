from itertools import izip


class SgdOptimizer(object):
    def __init__(self, learning_rate, model):
        self.learning_rate = learning_rate
        self.model = model
        self.param_u_contexts, self.params = zip(*self.model.params)
        self.grad_o_contexts, self.grads = zip(*self.model.grads)

    def optimize(self):
        while True:
            self.update()

    def update(self):
        for param_u_context, param, grad_o_context, grad in izip(self.param_u_contexts, self.params, self.grad_o_contexts, self.grads):
            param_u_context.wait(grad_o_context)
            if type(grad) is tuple:
                param.sliced_add_scaled(param_u_context, grad[0], self.learning_rate, grad[1])
            else:
                param.add_scaled(param_u_context, self.learning_rate, grad)