from quagga.matrix import Matrix
from collections import defaultdict
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

    def __init__(self, f_matrix, f_obtaining_context, b_usage_context=None):
        self._f_matrices = {f_obtaining_context.device_id: f_matrix}
        self._f_obtaining_context = f_obtaining_context
        self._f_usage_contexts = list()
        if b_usage_context:
            self._b_matrices = dict()
            self._b_usage_context = b_usage_context
            self._b_matrices_pool = defaultdict(dict)
            self._last_b_obtaining_context = None

    @property
    def bpropagable(self):
        return bool(self._b_usage_context)

    def register_usage(self, f_usage_context, b_obtaining_context=None, accumulation_method_name=None, is_sparse=False):
        """
        Register usage of connector's forward_matrix.

        :param f_usage_context: context in which `forward_matrix` will be used
        :param b_obtaining_context: context in which `backward_matrix`
                                    of the connector will be calculated
        """

        if not self._b_usage_context and b_obtaining_context:
            raise ValueError('Nobody is going to use computation from backward '
                             'step. You should not backward propagate!')
        fu_device_id = f_usage_context.device_id
        fo_device_id = self._f_obtaining_context.device_id
        if fu_device_id != fo_device_id and fu_device_id not in self._f_matrices:
            self._f_matrices[fu_device_id] = Matrix.empty_like(self, fu_device_id)
        self._f_usage_contexts.append(f_usage_context)
        if not self._b_usage_context:
            return self._f_matrices[fu_device_id]

        bu_device_id = self._b_usage_context.device_id
        bo_device_id = b_obtaining_context.device_id

        if (bo_device_id, is_sparse) not in self._b_matrices:
            if is_sparse:
                bwd_m = SparseMatrix(bo_device_id)
            else:
                bwd_m = Matrix.empty_like(self, bo_device_id)
            self._b_matrices[bo_device_id, is_sparse] = bwd_m
            accumulation_method = getattr(bwd_m, accumulation_method_name)
            setattr(bwd_m, accumulation_method_name, self.decorate_waiting(accumulation_method))

        if bu_device_id != bo_device_id and (bu_device_id, is_sparse) not in self._b_matrices_pool:
            if is_sparse:
                bwd_m = SparseMatrix(bu_device_id)
            else:
                bwd_m = Matrix.empty_like(self, bu_device_id)
            self._b_matrices_pool[bu_device_id, is_sparse] = bwd_m

        return self._f_matrices[fu_device_id], self._b_matrices[bo_device_id, is_sparse]

    def fprop(self):
        o_device_id = self._f_obtaining_context.device_id
        for u_device_id, forward_matrix in self._f_matrices.iteritems():
            if u_device_id != o_device_id:
                self._f_matrices[o_device_id].copy_to(self._f_obtaining_context, forward_matrix)
        self._f_obtaining_context.block(*self._f_usage_contexts)

    def bprop(self):
        if not self._b_usage_context:
            raise ValueError('Nobody was going to use computation from backward '
                             'step. You should not backward propagate!')
        self._last_b_obtaining_context.block(self._b_usage_context)
        u_device_id = self._b_usage_context.device_id

        for o_device_id in set(zip(*self._b_matrices.keys())[0]):
            bwd_dense = self._b_matrices.get((o_device_id, False))
            bwd_sparse = self._b_matrices.get((o_device_id, True))
            if bwd_dense:
                if bwd_sparse:
                    bwd_dense.add(bwd_sparse)
                if u_device_id != o_device_id:
                    bwd_dense.copy_to(self._b_usage_context, self._b_matrices_pool[u_device_id, False])
                self._b_matrices[o_device_id, False].add(self._b_usage_context, self._b_matrices_pool[u_device_id, False])
                bwd = self._b_matrices[u_device_id, False]
            else:
                if u_device_id != o_device_id:
                    bwd_sparse.copy_to(self._b_usage_context, self._b_matrices_pool[u_device_id, True])
                self._b_matrices[o_device_id, True].add(self._b_usage_context, self._b_matrices_pool[u_device_id, True])
                bwd = self._b_matrices[u_device_id, True]
        self._last_b_obtaining_context = None
        return bwd

    backward_matrix = property(lambda self: self.bprop())

    def __getattr__(self, name):
        forward_matrix = self._f_matrices[self._f_obtaining_context.device_id]
        attribute = getattr(forward_matrix, name)
        if hasattr(attribute, '__call__'):
            setattr(self, name, attribute)
        else:
            setattr(Connector, name, property(lambda self: getattr(forward_matrix, name)))
        return getattr(self, name)

    def decorate_waiting(self, accumulation_method):
        def decorated_accumulation_method(context, *args, **kwargs):
            if self._last_b_obtaining_context:
                context.wait(self._last_b_obtaining_context)
            self._last_b_obtaining_context = context
            return accumulation_method(context, args, kwargs)
        return decorated_accumulation_method