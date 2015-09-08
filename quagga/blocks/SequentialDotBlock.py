import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.matrix import MatrixList
from quagga.connector import Connector


class SequentialDotBlock(object):
    def __init__(self, W_init, b_init, x_sequence, learning=True, device_id=None):
        """
        TODO
        """
        if not all(e.bpropagable for e in x_sequence) and \
                not all(not e.bpropagable for e in x_sequence):
            raise ValueError('All elements of x should be bpropagable '
                             'or non-bpropagable. Mixed state is not allowed!')
        self.W = Matrix.from_npa(W_init(), device_id=device_id)
        if learning:
            self.dL_dW = Matrix.empty_like(self.W)
            self.grads_context = Context(device_id)
        if b_init:
            self.b = Matrix.from_npa(b_init(), device_id=device_id)
            if learning:
                self.dL_db = Matrix.empty_like(self.b)
        self.dot_blocks = []
        for x in x_sequence:
            if learning:
                dot_block = _DotBlock(self.W, self.b, x, device_id, self.dL_dW, self.dL_db, self.grads_context)
            else:
                dot_block = _DotBlock(self.W, self.b, x, device_id)
            self.dot_blocks.append(dot_block)
        self.x_sequence = x_sequence
        self.max_input_sequence_len = len(x_sequence)
        self.output_sequence = MatrixList([e.output for e in self.dot_blocks])

    def fprop(self):
        n = len(self.x_sequence)
        if n > self.max_input_sequence_len:
            raise ValueError('Sequence has length: {} that is too long. '
                             'The maximum is: {}'.
                             format(n, self.max_input_sequence_len))
        self.output_sequence.set_length(n)
        for e in self.dot_blocks[:n]:
            e.fprop()

    def bprop(self):
        self.dL_dW.fill(self.grads_context, 0.0)
        if hasattr(self, 'dL_db'):
            self.dL_db.fill(self.grads_context, 0.0)
        n = len(self.x_sequence)
        for e in reversed(self.dot_blocks[:n]):
            e.bprop()

    @property
    def params(self):
        if hasattr(self, 'b'):
            return [(self.grads_context, self.W), (self.grads_context, self.b)]
        else:
            return [(self.grads_context, self.W)]

    @property
    def grads(self):
        if hasattr(self, 'dL_db'):
            return [(self.grads_context, self.dL_dW), (self.grads_context, self.dL_db)]
        else:
            return [(self.grads_context, self.dL_dW)]

    def get_parameter_initializers(self):
        initializers = {'W_init': lambda: self.W.to_host()}
        if hasattr(self, 'b'):
            initializers['b_init'] = lambda: self.b.to_host()
        return initializers


class _DotBlock(object):
    def __init__(self, W, b, x, device_id, dL_dW=None, dL_db=None, grads_context=None):
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.W = W
        if b:
            self.b = b
        learning = False
        if dL_dW:
            learning = True
            self.grads_context = grads_context
            self.dL_dW = dL_dW
        if dL_db:
            self.dL_db = dL_db
            self.ones = Matrix.from_npa(np.ones((x.nrows, 1), dL_db.np_dtype), device_id=device_id)
        self.learning = learning
        output = Matrix.empty(x.nrows, self.W.ncols, device_id=device_id)
        self.output = Connector(output, self.context, self.context if learning else None)
        if learning and x.bpropagable:
            self.x, self.dL_dx = x.register_usage(self.context, self.context)
        else:
            self.x = x.register_usage(self.context)

    def fprop(self):
        self.output.assign_dot(self.context, self.x, self.W)
        if hasattr(self, 'b'):
            self.output.add(self.context, self.b)
        self.output.fprop()

    def bprop(self):
        dL_doutput = self.output.backward_matrix
        # dL/dW = x.T * dL_doutput
        self.dL_dW.assign_dot(self.grads_context, self.x, dL_doutput, 'T')
        # dL/db = 1.T * error
        if hasattr(self, 'dL_db'):
            self.dL_db.assign_dot(self.grads_context, self.ones, dL_doutput, 'T')
        # dL/dx = dL_doutput * W.T
        if hasattr(self, 'dL_dx'):
            self.dL_dx.assign_dot(self.context, dL_doutput, self.W, 'N', 'T')