# ----------------------------------------------------------------------------
# Copyright 2015 Grammarly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
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
        self.context = {self._fo_device_id: Context(self._fo_device_id)}
        if bu_device_id is not None:
            self._bu_device_id = bu_device_id
            self._b_matrices = dict()
            self._b_matrices_pool = dict()
            self._b_sparse_matrix = None
        # We need do this trick because instead we will add attribute
        # to the Connector instance by setting it
        # instead of setting attribute in f_matrix
        self.__f_matrix_setable_attributes = f_matrix.get_setable_attributes()
        for attr_name in self.__f_matrix_setable_attributes:
            getattr(self, attr_name)

    @property
    def bpropagable(self):
        return hasattr(self, '_bu_device_id')

    def register_usage_with_sparse_backward_matrix(self):
        if self._bu_device_id != self._fo_device_id:
            raise ValueError("Registering usage with sparse backward matrix "
                             "requires equal forward obtaining device and "
                             "backward usage device.")
        fwd_matrix = self._f_matrices[self._fo_device_id]
        if self._b_sparse_matrix:
            return fwd_matrix, self._b_sparse_matrix
        self._b_sparse_matrix = SparseMatrix()
        return fwd_matrix, self._b_sparse_matrix

    def register_usage(self, fu_device_id, bo_device_id=None):
        """
        Register usage of connector's forward_matrix.

        :param fu_device_id: context in which `forward_matrix` will be used
        :param bo_device_id: context in which `backward_matrix`
                                    of the connector will be calculated
        """

        if not self.bpropagable and bo_device_id:
            raise ValueError("Nobody is going to use computation from backward step. "
                             "You mustn't register for backward propagate!")
        if fu_device_id != self._fo_device_id and fu_device_id not in self._f_matrices:
            self._f_matrices[fu_device_id] = Matrix.empty_like(self, fu_device_id)
            self.context[fu_device_id] = Context(fu_device_id)
        if bo_device_id is None:
            return self._f_matrices[fu_device_id]

        for device_id in [self._bu_device_id, bo_device_id]:
            if device_id not in self._b_matrices:
                self._b_matrices[device_id] = Matrix.empty_like(self, device_id)
                if device_id not in self.context:
                    self.context[device_id] = Context(device_id)
        if self._bu_device_id != bo_device_id and self._bu_device_id not in self._b_matrices_pool:
            self._b_matrices_pool[self._bu_device_id] = Matrix.empty_like(self, self._bu_device_id)
        return self._f_matrices[fu_device_id], self._b_matrices[bo_device_id]

    def fprop(self):
        for u_device_id, forward_matrix in self._f_matrices.iteritems():
            if u_device_id != self._fo_device_id:
                forward_matrix.assign(self.context[u_device_id], self._f_matrices[self._fo_device_id])

        if self.bpropagable:
            for bo_device_id, matrix in self._b_matrices.iteritems():
                if bo_device_id == self._bu_device_id and matrix.last_usage_context:
                    # one must use last_usage_context otherwise we could be
                    # modifying matrix while updating parameters or
                    # propagating derivatives.
                    context = matrix.last_usage_context
                else:
                    context = self.context[matrix.device_id]
                matrix.fill(context, 0.0)
            if self._b_sparse_matrix:
                self._b_sparse_matrix.clear()

    def bprop(self):
        if not self.bpropagable:
            raise ValueError('Nobody was going to use computation from backward '
                             'step. You should not backward propagate!')
        if not self._b_matrices and not self._b_sparse_matrix:
            # When no one registered for providing derivatives zero dense
            # matrix will be returned
            bwd = Matrix.empty_like(self, self._bu_device_id)
            if self._bu_device_id not in self.context:
                self.context[self._bu_device_id] = Context(self._bu_device_id)
            bwd.fill(self.context[self._bu_device_id], 0.0)
            self._b_matrices[self._bu_device_id] = bwd
            return bwd

        if not self._b_matrices and self._b_sparse_matrix:
            return self._b_sparse_matrix

        for bo_device_id, bwd_matrix in self._b_matrices.iteritems():
            if self._bu_device_id != bo_device_id:
                self._b_matrices_pool[self._bu_device_id].assign(self.context[self._bu_device_id], bwd_matrix)
                self._b_matrices[self._bu_device_id].add(self.context[self._bu_device_id], self._b_matrices_pool[self._bu_device_id])
        if self._b_sparse_matrix:
            self._b_matrices[self._bu_device_id].add(self.context[self._bu_device_id], self._b_sparse_matrix)
        return self._b_matrices[self._bu_device_id]

    backward_matrix = property(lambda self: self.bprop())

    def __getattr__(self, name):
        attribute = getattr(self._f_matrices[self._fo_device_id], name)
        if hasattr(attribute, '__call__'):
            setattr(self, name, attribute)
        else:
            # TODO(sergii): inspect this place. hyp: property belong to class not to instance
            fget = lambda self: getattr(self._f_matrices[self._fo_device_id], name)
            if name in self.__f_matrix_setable_attributes:
                fset = lambda self, value: setattr(self._f_matrices[self._fo_device_id], name, value)
                setattr(Connector, name, property(fget, fset))
            else:
                setattr(Connector, name, property(fget))
        return getattr(self, name)

    def __getitem__(self, item):
        return self._f_matrices[self._fo_device_id][item]