from quagga.matrix import Matrix
from quagga.context import Context
from quagga.matrix import SparseMatrix


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

    def __init__(self, f_matrix, bu_device_id=None):
        self._fo_device_id = f_matrix.device_id
        self._f_matrices = {self._fo_device_id: f_matrix}
        if bu_device_id is not None:
            self._bu_device_id = bu_device_id
            self._b_matrices = dict()
            # self._b_matrices_pool = dict()
        self.context = Context(self._fo_device_id)

    @property
    def bpropagable(self):
        return hasattr(self, '_bu_device_id')

    def register_usage(self, fu_device_id, bo_device_id=None, is_sparse=False):
        """
        Register usage of connector's forward_matrix.

        :param fu_device_id: context in which `forward_matrix` will be used
        :param bo_device_id: context in which `backward_matrix`
                                    of the connector will be calculated
        """

        if not self.bpropagable and bo_device_id:
            raise ValueError('Nobody is going to use computation from backward '
                             'step. You should not backward propagate!')
        if fu_device_id != self._fo_device_id and fu_device_id not in self._f_matrices:
            self._f_matrices[fu_device_id] = Matrix.empty_like(self, fu_device_id)
        if not self.bpropagable:
            return self._f_matrices[fu_device_id]

        if (bo_device_id, is_sparse) not in self._b_matrices:
            if is_sparse:
                bwd_m = SparseMatrix(bo_device_id)
            else:
                bwd_m = Matrix.empty_like(self, bo_device_id)
            self._b_matrices[bo_device_id, is_sparse] = bwd_m
        if self._bu_device_id != bo_device_id and (self._bu_device_id, is_sparse) not in self._b_matrices_pool:
            if is_sparse:
                bwd_m = SparseMatrix(self._bu_device_id)
            else:
                bwd_m = Matrix.empty_like(self, self._bu_device_id)
            self._b_matrices_pool[self._bu_device_id, is_sparse] = bwd_m
        return self._f_matrices[fu_device_id], self._b_matrices[bo_device_id, is_sparse]

    def fprop(self):
        o_device_id = self._fo_device_id
        for u_device_id, forward_matrix in self._f_matrices.iteritems():
            if u_device_id != o_device_id:
                forward_matrix.assign(self.context, self._f_matrices[o_device_id])

        if self.bpropagable:
            for (o_device_id, is_sparse), matrix in self._b_matrices.iteritems():
                if is_sparse:
                    matrix.clear()
                else:
                    matrix.fill(self.context, 0.0)

    def bprop(self):
        if not self.bpropagable:
            raise ValueError('Nobody was going to use computation from backward '
                             'step. You should not backward propagate!')
        u_device_id = self._bu_device_id

        for o_device_id in set(zip(*self._b_matrices.keys())[0]):
            bwd_dense = self._b_matrices.get((o_device_id, False))
            bwd_sparse = self._b_matrices.get((o_device_id, True))
            if bwd_dense:
                if bwd_sparse:
                    # TODO(sergii) bwd_dense.add(context, bwd_sparse)
                    pass
                if u_device_id != o_device_id:
                    self._b_matrices_pool[u_device_id, False].assign(self.context, bwd_dense)
                    self._b_matrices[u_device_id, False].add(self.context, self._b_matrices_pool[u_device_id, False])
            else:
                if u_device_id != o_device_id:
                    bwd_sparse.copy_to(self.context, self._b_matrices_pool[u_device_id, True])
                self._b_matrices[o_device_id, True].add(self._b_matrices_pool[u_device_id, True])

        if (u_device_id, False) in self._b_matrices:
            return self._b_matrices[u_device_id, False]
        else:
            return self._b_matrices[u_device_id, True]

    backward_matrix = property(lambda self: self.bprop())

    def __getattr__(self, name):
        attribute = getattr(self._f_matrices[self._fo_device_id], name)
        if hasattr(attribute, '__call__'):
            setattr(self, name, attribute)
        else:
            setattr(Connector, name, property(lambda self: getattr(self._f_matrices[self._fo_device_id], name)))
        return getattr(self, name)