from quagga.matrix import Matrix


class Connector(object):
    """
    Instance of `Connector` class is aimed to connect blocks.
    It has all the functionalities that `Matrix` instance has.
    All `Matrix`-related methods and variables are delegated and taken from
    `forward_matrix` variable, so you can use it as a `Matrix` instance
    without any extra actions during block's `fprop`.
    """

    def __init__(self, forward_matrix, forward_context=None):
        self.forward_matrix = forward_matrix
        self.forward_context = forward_context
        self._backward_matrices = dict()
        self._backward_contexts = dict()

    def register_user(self, user, backward_context, backward_matrix=None):
        """
        Register user of connector's forward_matrix.

        :param user: block that will use this connector
        :param backward_context: context in which `backward_matrix` of the
                                 connector will be calculated
        :param backward_matrix: backward_matrix buffer if it is None
                                the same as forward_matrix will be created
        """
        self._backward_contexts[user] = backward_context
        self._backward_matrices[user] = backward_matrix if backward_matrix else Matrix.empty_like(self.forward_matrix)

    def forward_block(self, context):
        """
        Block any calculation in `context` until all calculation in context
        in which connector's `forward_matrix` is calculating is done.
        """

        if self.forward_context:
            self.forward_context.block(context)

    def backward_block(self, context):
        """
        Block any calculation in `context` until all calculation in context
        in which connector's `backward_matrix` is calculating is done.
        """

        self._backward_contexts.values()[0].block(context)

    def get_backward_matrix(self, requester=None):
        if requester:
            return self._backward_matrices[requester]
        backward_matrices = self._backward_matrices.values()
        backward_contexts = self._backward_contexts.values()
        backward_contexts[0].depend_on(*backward_contexts[1:])
        for backward_matrix in backward_matrices[1:]:
            # TODO add sparse gradient handle
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
