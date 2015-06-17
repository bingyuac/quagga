from quagga.matrix import Matrix


class Connector(object):
    """
    Instance of `Connector` class is aimed to connect blocks.
    It has all the functionalities that `Matrix` instance has.
    All `Matrix`-related methods and variables are delegated and taken from
    `forward_matrix` variable, so you can use it as a `Matrix` instance
    without any extra actions during block's `fprop`. Final `backward_matrix`
    is forming by summation all `_backward_matrices` that is why they must have
    the same shapes and types.

                                +----------------------+
                                |      connector       |
                                |----------------------|    +----------------->
     _forward_obtaining_context |  +--------------+    +----+
    +-------------------------->|  |forward_matrix|    |_forward_usage_contexts
                                |  +--------------+    +----+
                                |                      |    +----------------->
       _backward_usage_context  |  +---------------+   |
    <---------------------------+  |backward_matrix|   |
                                |  +---------------+   |
                                |          ^           |
                                |          |           |    +-----------------+
                                | +--------+---------+ |<---+
                                | |_backward_matrices| |_backward_obtaining_contexts
                                | +------------------+ |<---+
                                +----------------------+    +-----------------+
    """

    def __init__(self, forward_matrix, forward_obtaining_context=None, backward_usage_context=None):
        self.forward_matrix = forward_matrix
        self._forward_obtaining_context = forward_obtaining_context
        self._forward_usage_contexts = dict()

        self._backward_matrices = dict()
        self._backward_obtaining_contexts = dict()
        self._backward_usage_context = backward_usage_context

    def register_user(self, user, forward_usage_context, backward_obtaining_context, backward_matrix=None):
        """
        Register user of connector's forward_matrix.

        :param user: block that will use this connector
        :param forward_usage_context: context in which `forward_matrix`
                                      will be used
        :param backward_obtaining_context: context in which `backward_matrix`
                                           of the connector will be calculated
        :param backward_matrix: backward_matrix buffer if it is None
                                the same as forward_matrix will be created
        """
        self._forward_usage_contexts[user] = forward_usage_context
        self._backward_obtaining_contexts[user] = backward_obtaining_context
        self._backward_matrices[user] = backward_matrix if backward_matrix else Matrix.empty_like(self.forward_matrix)

    def block_users(self):
        """
        From this moment all further calculations in user-registered contexts
        will be blocked until all calculations that were launched previously
        in the context where the `forward_matrix` is being calculated,
        is done.
        """

        if self._forward_obtaining_context:
            self.forward_context.block(self._forward_usage_contexts.itervalues())

    def depend_on_users(self):
        """
        From this moment all further calculations in the
        `_backward_usage_context` will be blocked until every calculations
        Blocks all calculations in `context` until all calculations that were
        launched previously in the `_backward_obtaining_contexts` is done.
        """

        if self._backward_usage_context:
            self._backward_usage_context.depend_on(self._backward_obtaining_contexts.itervalues())

    def get_backward_matrix(self, requester=None):
        if requester:
            return self._backward_matrices[requester]
        backward_matrices = self._backward_matrices.values()
        backward_contexts = self._backward_obtaining_contexts.values()
        backward_contexts[0].depend_on(*backward_contexts[1:])
        for backward_matrix in backward_matrices[1:]:
            backward_matrices[0].add(self.backward_context, backward_matrix)
        return backward_matrices[0]

    backward_matrix = property(get_backward_matrix)

    def __getattr__(self, name):
        attribute = getattr(self.forward_matrix, name)
        if hasattr(attribute, '__call__'):
            setattr(self, name, lambda *args, **kwargs: getattr(self.forward_matrix, name)(*args, **kwargs))
        else:
            fget = lambda self: getattr(self.matrix, name)
            fset = lambda self, value: setattr(self.matrix, name, value)
            setattr(Connector, name, property(fget, fset))
        return getattr(self, name)
