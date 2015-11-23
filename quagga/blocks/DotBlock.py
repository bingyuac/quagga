from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class DotBlock(object):
    def __init__(self, W, b, x, device_id=None):
        self.f_context = Context(device_id)
        device_id = self.f_context.device_id

        if W.bpropagable:
            self.W, self.dL_dW = W.register_usage(device_id, device_id)
        else:
            self.W = W.register_usage(device_id)
        if b:
            if b.bpropagable:
                self.b, self.dL_db = b.register_usage(device_id, device_id)
                self.ones = Matrix.empty(x.nrows, 1, self.b.dtype, device_id)
                self.ones.sync_fill(1.0)
            else:
                self.b = b.register_usage(device_id)
        if x.bpropagable:
            self.x, self.dL_dx = x.register_usage(device_id, device_id)
        else:
            self.x = x.register_usage(device_id)

        output = Matrix.empty(x.nrows, self.W.ncols, device_id=device_id)
        self.learning = hasattr(self, 'dL_dW') or hasattr(self, 'dL_db') or \
                        hasattr(self, 'dL_dx')
        if self.learning:
            self.b_context = Context(device_id)
            self.output = Connector(output, device_id)
        else:
            self.output = Connector(output)

    def fprop(self):
        self.output.assign_dot(self.f_context, self.x, self.W)
        if hasattr(self, 'b'):
            self.output.add(self.f_context, self.b)
        self.output.fprop()

    def bprop(self):
        if not self.learning:
            return
        dL_doutput = self.output.backward_matrix
        # dL/dW = x.T * dL_doutput
        if hasattr(self, 'dL_dW'):
            self.dL_dW.add_dot(self.b_context, self.x, dL_doutput, 'T')
        # TODO(sergii): replace this modification with reduction kernel along axis=0
        # dL/db = 1.T * dL_doutput
        if hasattr(self, 'dL_db'):
            self.dL_db.add_dot(self.b_context, self.ones, dL_doutput, 'T')
        # dL/dx = dL_doutput * W.T
        if hasattr(self, 'dL_dx'):
            self.dL_dx.add_dot(self.b_context, dL_doutput, self.W, 'N', 'T')