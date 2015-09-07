import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context


class SoftmaxCeBlock(object):
    """
    Softmax nonlinearity with mean cross entropy loss
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

        # TODO fix here true_labels may be one-hot coding or just numbers

        self.probs = Matrix.empty_like(true_labels, device_id=self.context.device_id)

        if self.true_labels.dtype == 'int':
            self.bprop = lambda self: self.dL_dx.assign_softmax_ce_derivative(self.context, self.probs, self.true_labels)
        else:
            self.bprop = lambda self: self.dL_dx.assign_scaled_subtraction(self.context, 1. / self.probs.nrows, self.probs, self.true_labels)

    def fprop(self):
        self.x.softmax(self.context, self.probs)

    def bprop(self):
        # error = (probs - true_labels) / M
        pass

    @property
    def loss(self):
        true_labels = self.true_labels.to_host()
        probs = self.probs.to_host()
        if self.true_labels.dtype == 'int':
            return - np.mean(np.log(probs[range(probs.shape[0]), true_labels.flatten()] + 1e-20))
        else:
            return - np.mean(np.sum(true_labels * np.log(probs + 1e-20), axis=1))