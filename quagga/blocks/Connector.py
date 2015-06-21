from quagga.matrix import Matrix
from collections import defaultdict


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
                                |  +-----------------+ +----+
                                |  |_forward_matrices| |_forward_usage_contexts
                                |  +-----------------+ +----+
                                |           ^          |    +----------------->
                                |           |          |
     _forward_obtaining_context |   +-------+------+   |
    +-------------------------->|   |forward_matrix|   |
                                |   +--------------+   |
                                |                      |
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

    def __init__(self, f_matrix, f_obtaining_context=None, b_usage_context=None):
        self._f_matrices = {f_obtaining_context.device_id: f_matrix}
        self._f_obtaining_context = f_obtaining_context
        self._f_usage_contexts = dict()

        self._b_matrices = defaultdict(dict)
        self._b_obtaining_contexts = dict()
        self._b_usage_context = b_usage_context

    def register_user(self, user, f_usage_context, b_obtaining_context, b_matrix=None):
        """
        Register user of connector's forward_matrix.

        :param user: block that will use this connector
        :param f_usage_context: context in which `forward_matrix`
                                      will be used
        :param b_obtaining_context: context in which `backward_matrix`
                                           of the connector will be calculated
        :param b_matrix: backward_matrix buffer if it is None
                                the same as forward_matrix will be created
        """
        u_device_id = f_usage_context.device_id
        o_device_id = self._f_obtaining_context.device_id
        if u_device_id != o_device_id and u_device_id not in self._f_matrices:
            self._f_matrices[u_device_id] = Matrix.empty_like(self._f_matrices[o_device_id], u_device_id)
        self._f_usage_contexts[user] = f_usage_context

        u_device_id = self._b_usage_context.device_id
        o_device_id = b_obtaining_context.device_id
        b_matrix = b_matrix if b_matrix else Matrix.empty_like(self.forward_matrix, o_device_id)
        if u_device_id != o_device_id:
            self._b_matrices[user][u_device_id] = Matrix.empty_like(b_matrix, u_device_id)
        self._b_matrices[user][o_device_id] = b_matrix
        self._b_obtaining_contexts[user] = b_obtaining_context

    def get_forward_matrix(self, requester=None):
        o_device_id = self._f_obtaining_context.device_id
        if not requester:
            return self._f_matrices[o_device_id]
        u_device_id = self._f_usage_contexts[requester].device_id
        if o_device_id == u_device_id:
            return self._f_matrices[o_device_id]
        self._f_matrices[o_device_id].copy(self._f_obtaining_context, self._f_matrices[u_device_id])
        self._f_usage_contexts[requester].wait(self._f_obtaining_context)
        return self._f_matrices[u_device_id]

    forward_matrix = property(get_forward_matrix)

    def get_backward_matrix(self, requester=None):
        if requester:
            o_device_id = self._b_obtaining_contexts[requester].device_id
            return self._b_matrices[requester][o_device_id]
        u_device_id = self._b_usage_context.device_id
        backward_matrices = []
        for requester, matrices in self._b_matrices.iteritems():
            b_obtaining_context = self._b_obtaining_contexts[requester]
            o_device_id = b_obtaining_context.device_id
            if u_device_id != o_device_id:
                matrices[o_device_id].copy(b_obtaining_context, matrices[u_device_id])
                self._b_usage_context.wait(b_obtaining_context)
            backward_matrices.append(matrices[u_device_id])
        for backward_matrix in backward_matrices[1:]:
            backward_matrices[0].add(self._b_usage_context, backward_matrix)
        return backward_matrices[0]

    backward_matrix = property(get_backward_matrix)

    def block_users(self):
        """
        From this moment all further calculations in user-registered contexts
        will be blocked until all calculations that were launched previously
        in the context where the `forward_matrix` is being calculated,
        is done.
        """

        if self._f_obtaining_context:
            self.forward_context.block(self._f_usage_contexts.itervalues())

    def wait_users(self):
        """
        From this moment all further calculations in the
        `_backward_usage_context` will be blocked until all calculations
        that were launched previously in the `_backward_obtaining_contexts`
        is done.
        """

        if self._b_usage_context:
            self._b_usage_context.wait(self._b_obtaining_contexts.itervalues())

    def __getattr__(self, name):
        attribute = getattr(self.forward_matrix, name)
        if hasattr(attribute, '__call__'):
            setattr(self, name, lambda *args, **kwargs: getattr(self.forward_matrix, name)(*args, **kwargs))
        else:
            fget = lambda self: getattr(self.matrix, name)
            fset = lambda self, value: setattr(self.matrix, name, value)
            setattr(Connector, name, property(fget, fset))
        return getattr(self, name)
