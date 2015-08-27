from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class ScaleBlock(object):
    def __init__(self, x, scale_factor, device_id=None):
        self.scale_factor = scale_factor
        self.context = Context(device_id)
        if x.bpropagable:
            self.x, self.dL_dx = x.register_usage(self.context, self.context)
        else:
            self.x = x.register_usage(self.context)
        self.output = Matrix.empty_like(x, self.context.device_id)
        self.output = Connector(self.output, self.context, self.context if x.bpropagable else None)

    def fprop(self):
        self.x.scale(self.context, self.scale_factor, self.output)
        self.output.fprop()

    def bprop(self):
        dL_doutput = self.output.backward_matrix
        dL_doutput.scale(self.context, self.scale_factor, self.dL_dx)

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []