from quagga.matrix import Matrix


class Connector(object):
    def __init__(self, matrix, forward_context=None):
        self.matrix = matrix
        self.forward_context = forward_context
        self._derivatives = dict()
        self._backward_contexts = dict()

    def register_user(self, user, backward_context, derivative=None):
        self._backward_contexts[user] = backward_context
        self._derivatives[user] = derivative if derivative else Matrix.empty_like(self.matrix)

    def block(self, context):
        if self.forward_context:
            self.forward_context.block(context)

    def backward_block(self, context):
        self._backward_contexts.values()[0].block(context)

    def get_derivative(self, requester=None):
        if requester:
            return self._derivatives[requester]
        derivatives = self._derivatives.values()
        backward_contexts = self._backward_contexts.values()
        backward_contexts[0].depend_on(*backward_contexts[1:])
        for derivative in derivatives[1:]:
            # TODO add sparse gradient handle
            derivatives[0].add(self.backward_context, derivative)
        return derivatives[0]

    derivative = property(get_derivative)

    def __getattr__(self, name):
        attribute = getattr(self.matrix, name)
        if hasattr(attribute, '__call__'):
            setattr(self, name, lambda *args, **kwargs: getattr(self.matrix, name)(*args, **kwargs))
        else:
            fget = lambda self: getattr(self.matrix, name)
            fset = lambda self, value: setattr(self.matrix, name, value)
            setattr(Connector, name, property(fget, fset))
        return getattr(self, name)
