import ctypes as ct
from quagga.context import Context


class L2RegularizationBlock(object):
    def __init__(self, W, regularization_value):
        self.context = Context(W.device_id)
        device_id = self.context.device_id
        if W.bpropagable:
            self.W, self.dL_dW = W.register_usage(device_id, device_id)
        else:
            self.W = W.register_usage(device_id)
        self.reg_value = ct.c_float(2 * regularization_value)

    def bprop(self):
        self.dL_dW.add_scaled(self, self.context, self.reg_value, self.W)