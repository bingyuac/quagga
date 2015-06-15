from quagga.matrix import Matrix


class Connector(object):
    def __init__(self, forward_matrix, forward_context=None):
        self.forward_matrix = forward_matrix
        self.forward_context = forward_context
        self._backward_matrices = dict()
        self._backward_contexts = dict()

    def register_user(self, user, backward_context, backward_matrix=None):
        """

        :param user: block that will be use this connector
        :param backward_context: context in which derivative of the target
                                 function wrt this connector will be calculated
        :param backward_matrix:
        """
        self._backward_contexts[user] = backward_context
        self._backward_matrices[user] = backward_matrix if backward_matrix else Matrix.empty_like(self.forward_matrix)

    def block(self, context):
        if self.forward_context:
            self.forward_context.block(context)

    def backward_block(self, context):
        self._backward_contexts.values()[0].block(context)

    def get_derivative(self, requester=None):
        if requester:
            return self._backward_matrices[requester]
        derivatives = self._backward_matrices.values()
        backward_contexts = self._backward_contexts.values()
        backward_contexts[0].depend_on(*backward_contexts[1:])
        for derivative in derivatives[1:]:
            # TODO add sparse gradient handle
            derivatives[0].add(self.backward_context, derivative)
        return derivatives[0]

    derivative = property(get_derivative)

    def __getattr__(self, name):
        attribute = getattr(self.forward_matrix, name)
        if hasattr(attribute, '__call__'):
            setattr(self, name, lambda *args, **kwargs: getattr(self.forward_matrix, name)(*args, **kwargs))
        else:
            fget = lambda self: getattr(self.matrix, name)
            fset = lambda self, value: setattr(self.matrix, name, value)
            setattr(Connector, name, property(fget, fset))
        return getattr(self, name)
