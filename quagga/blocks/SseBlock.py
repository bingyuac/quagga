import numpy as np
from quagga.context import Context


class SseBlock(object):
    """
    Sum of squared of the errors
    """

    def __init__(self, y_hat, y, device_id=None):
        if y_hat.nrows != y.nrows or y_hat.ncols != y.ncols:
            raise ValueError('TODO!')
        self.context = Context(device_id)
        if y_hat.bpropagable:
            self.y_hat, self.dL_dy_hat = y_hat.register_usage(self.context, self.context)
        else:
            self.y_hat = y_hat.register_usage(self.context)
        self.y = y.register_usage(self.context)

    def fprop(self):
        pass

    def bprop(self):
        # error = (y_hat - y) / M
        self.dL_dy_hat.add_scaled_subtraction(self.context, 2. / self.y.nrows, self.y_hat, self.y)

    @property
    def loss(self):
        y = self.y.to_host()
        y_hat = self.y_hat.to_host()
        return np.sum((y - y_hat) ** 2) / y.shape[0]

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []