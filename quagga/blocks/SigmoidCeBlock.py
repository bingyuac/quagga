import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context


class SigmoidCeBlock(object):
    """
    Sigmoid nonlinearity with mean cross entropy loss
    """

    def __init__(self, x, true_labels, device_id=None):
        if x.nrows != true_labels.nrows:
            raise ValueError('TODO!')
        self.context = Context(device_id)
        if x.bpropagable:
            self.x, self.dL_dx = x.register_usage(self.context, self.context)
        else:
            self.x = x.register_usage(self.context)
        self.true_labels = true_labels.register_usage(self.context)
        self.probs = Matrix.empty_like(true_labels, device_id=self.context.device_id)

    def fprop(self):
        self.x.sigmoid(self.context, self.probs)

    def bprop(self):
        # error = (probs - true_labels) / M
        self.dL_dx.assign_scaled_subtraction(self.context, 1. / self.probs.nrows, self.probs, self.true_labels)

    @property
    def loss(self):
        true_labels = self.true_labels.to_host()
        probs = self.probs.to_host()
        return - (true_labels * np.log(probs + 1e-20) +
                  (1.0 - true_labels) * np.log(1. - probs + 1e-20))

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []