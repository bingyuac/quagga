import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class DotBlock(object):
    def __init__(self, W_init, b_init, x, learning=True, device_id=None):
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.W = Matrix.from_npa(W_init(), device_id=device_id)
        if b_init:
            self.b = Matrix.from_npa(b_init(), device_id=device_id)
        output = Matrix.empty(x.nrows, self.W.ncols, device_id=device_id)
        self.output = Connector(output, self.context, self.context if learning else None)
        if learning:
            self.dL_dW = Matrix.empty_like(self.W)
            if hasattr(self, 'b'):
                self.dL_db = Matrix.empty_like(self.b)
                self.ones = Matrix.from_npa(np.ones((x.nrows, 1), self.b.np_dtype), device_id=device_id)
            if x.bpropagable:
                self.x, self.dL_dx = x.register_usage(self.context, self.context)
            else:
                self.x = x.register_usage(self.context)
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
        self.dL_dW.assign_dot(self.context, self.x, dL_doutput, 'T')
        # dL/db = 1.T * error
        if hasattr(self, 'dL_db'):
            self.dL_db.assign_dot(self.context, self.ones, dL_doutput, 'T')
        # dL/dx = dL_doutput * W.T
        if hasattr(self, 'dL_dx'):
            self.dL_dx.assign_dot(self.context, dL_doutput, self.W, 'N', 'T')

    @property
    def params(self):
        if hasattr(self, 'b'):
            return [(self.context, self.W), (self.context, self.b)]
        else:
            return [(self.context, self.W)]

    @property
    def grads(self):
        if hasattr(self, 'dL_db'):
            return [(self.context, self.dL_dW), (self.context, self.dL_db)]
        else:
            return [(self.context, self.dL_dW)]

    def get_parameter_initializers(self):
        initializers = {'W_init': lambda: self.W.to_host()}
        if hasattr(self, 'b'):
            initializers['b_init'] = lambda: self.b.to_host()
        return initializers