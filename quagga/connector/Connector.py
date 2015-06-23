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

    def __init__(self, f_matrix, f_obtaining_context, b_usage_context=None):
        self._f_matrices = {f_obtaining_context.device_id: f_matrix}
        self._f_obtaining_context = f_obtaining_context
        self._f_usage_contexts = list()
        self._b_matrices = defaultdict(dict)
        self._b_obtaining_contexts = dict()
        self._b_usage_context = b_usage_context

    def register_usage(self, f_usage_context, b_obtaining_context=None):
        """
        Register user of connector's forward_matrix.

        :param f_usage_context: context in which `forward_matrix`
                                      will be used
        :param b_obtaining_context: context in which `backward_matrix`
                                    of the connector will be calculated
        """
        # if not self._b_usage_context and b_obtaining_context:
        #     raise ValueError('Why do you perform backward propagation? '
        #                      'Previous block does not need you backward step')

        u_device_id = f_usage_context.device_id
        o_device_id = self._f_obtaining_context.device_id
        if u_device_id != o_device_id and u_device_id not in self._f_matrices:
            self._f_matrices[u_device_id] = Matrix.empty_like(self, u_device_id)
        self._f_usage_contexts.append(f_usage_context)
        if not self._b_usage_context:
            return self._f_matrices[f_usage_context.device_id]

        u_device_id = self._b_usage_context.device_id
        o_device_id = b_obtaining_context.device_id
        b_matrix = Matrix.empty_like(self, o_device_id)
        if u_device_id != o_device_id:
            self._b_matrices[b_obtaining_context][u_device_id] = Matrix.empty_like(b_matrix, u_device_id)
        self._b_matrices[b_obtaining_context][o_device_id] = b_matrix
        return self._f_matrices[f_usage_context.device_id], b_matrix

    def fprop(self):
        o_device_id = self._f_obtaining_context.device_id
        for u_device_id, forward_matrix in self._f_matrices.iteritems():
            if u_device_id != o_device_id:
                self._f_matrices[o_device_id].copy(self._f_obtaining_context, forward_matrix)
        self._f_obtaining_context.block(*self._f_usage_contexts)

    def bprop(self):
        u_device_id = self._b_usage_context.device_id
        backward_matrices = []
        for b_obtaining_context, matrices in self._b_matrices.iteritems():
            o_device_id = b_obtaining_context.device_id
            if u_device_id != o_device_id:
                matrices[o_device_id].copy(b_obtaining_context, matrices[u_device_id])
            backward_matrices.append(matrices[u_device_id])
        self._b_usage_context.wait(*self._b_matrices.iterkeys())
        for backward_matrix in backward_matrices[1:]:
            backward_matrices[0].add(self._b_usage_context, backward_matrix)
        return backward_matrices[0]

    backward_matrix = property(bprop)

    def __getattr__(self, name):
        attribute = getattr(self._f_matrices[self._f_obtaining_context.device_id], name)
        if hasattr(attribute, '__call__'):
            setattr(self, name, attribute)
        else:
            setattr(Connector, name, property(lambda self: getattr(self._f_matrices[self._f_obtaining_context.device_id], name)))
        return getattr(self, name)

    @property
    def nrows(self):
        return self._f_matrices[self._f_obtaining_context.device_id].nrows

    @nrows.setter
    def nrows(self, value):
        for forward_matrix in self._f_matrices.itervalues():
            forward_matrix.nrows = value
        for matrices in self._b_matrices.itervalues():
            for matrix in matrices.itervalues():
                matrix.nrows = value

    @property
    def ncols(self):
        return self._f_matrices[self._f_obtaining_context.device_id].ncols

    @ncols.setter
    def ncols(self, value):
        for forward_matrix in self._f_matrices.itervalues():
            forward_matrix.ncols = value
        for matrices in self._b_matrices.itervalues():
            for matrix in matrices.itervalues():
                matrix.ncols = value