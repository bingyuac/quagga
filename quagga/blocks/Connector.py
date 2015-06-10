from quagga.matrix import Matrix


class Connector(object):
    def __init__(self, matrix, forward_context=None):
        self.matrix = matrix
        self.forward_context = forward_context
        self.updated = False
        self._derivatives = dict()
        self._backward_contexts = dict()

    def register_user(self, user, backward_context):
        self._derivatives[user] = Matrix.empty_like(self.matrix)
        self._backward_contexts[user] = backward_context

    def block(self, context):
        if self.forward_context:
            self.forward_context.block(context)

    def backward_block(self, context):
        self._backward_contexts.values()[0].block(context)

    def get_derivative(self, requester=None):
        if requester:
            return self._derivatives[requester]
        # TODO investigate about update field maybe you should remove it
        derivatives = self._derivatives.values()
        if self.updated:
            backward_contexts = self._backward_contexts.values()
            backward_contexts[0].depend_on(*backward_contexts[1:])
            for derivative in derivatives[1:]:
                derivatives[0].add(self.backward_context, derivative)
            self.updated = False
        return derivatives[0]

    derivative = property(get_derivative)

    def __getattr__(self, name):
        attribute = getattr(self.matrix, name)
        if hasattr(attribute, '__call__'):
            def update_tracking_attribute(*args, **kwargs):
                self.updated = True
                return getattr(self.a, name)(*args, **kwargs)
            setattr(self, name, update_tracking_attribute)
        else:
            def fget(self):
                self.updated = True
                return getattr(self.matrix, name)

            def fset(self, value):
                self.updated = True
                setattr(self.matrix, name, value)
            setattr(Connector, name, property(fget, fset))
        return getattr(self, name)
