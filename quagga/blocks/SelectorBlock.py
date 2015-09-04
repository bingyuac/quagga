from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class SelectorBlock(object):
    def __init__(self, x, device_id=None):
        """
        TODO
        """
        if all(e.bpropagable for e in x):
            learning = True
        elif all(not e.bpropagable for e in x):
            learning = False
        else:
            raise ValueError('All elements of x should be bpropagable '
                             'or non-bpropagable. Mixed state is not allowed!')
        self.max_input_sequence_len = len(x)
        self.context = Context(device_id)
        self.output = Matrix.empty_like(x[0], self.context.device_id)
        self.output = Connector(self.output, self.context, self.context if learning else None)
        self._x = x
        self.x = []
        if learning:
            self.dL_dx = []
        for e in x:
            if learning:
                e, dL_de = e.register_usage(self.context, self.context)
                self.dL_dx.append(dL_de)
            else:
                e = e.register_usage(self.context)
            self.x.append(e)
        self.index = None

    def fprop(self, index):
        n = len(self._x)
        if n > self.max_input_sequence_len:
            raise ValueError('Sequence has length: {} that is too long. '
                             'The maximum is: {}'.
                             format(n, self.max_input_sequence_len))
        if index >= n:
            raise IndexError('Index: {} is out of range. Range is: [0, {})'.
                             format(index, n))
        if hasattr(self, 'dL_dx') and self.index is not None:
            for i, x in enumerate(self._x):
                if i != self.index:
                    x.remove_from_deregistered_b_obtaining_contexts(self.context)
        self.index = index
        self.x[index].copy_to(self.context, self.output)
        self.output.fprop()

    def bprop(self):
        for i, x in enumerate(self._x):
            if i != self.index:
                x.deregistere_b_obtaining_context(self.context)
        self.output.backward_matrix.copy_to(self.context, self.dL_dx[self.index])