import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context


class SoftmaxCeBlock(object):
    """
    Softmax nonlinearity with mean cross entropy loss
    """

    def __init__(self, x, true_labels, device_id=None):
        self.context = Context(device_id)
        device_id = self.context.device_id
        if x.bpropagable:
            self.x, self.dL_dx = x.register_usage(device_id, device_id)
        else:
            self.x = x.register_usage(device_id)
        self.true_labels = true_labels.register_usage(device_id)
        self.probs = Matrix.empty_like(self.x)
        self.true_labels_np = None
        self.probs_np = None
        # error = (probs - true_labels) / M
        if self.true_labels.dtype == 'int':
            self.bprop = lambda: self.dL_dx.assign_softmax_ce_derivative(self.context, self.probs, self.true_labels)
        else:
            self.bprop = lambda: self.dL_dx.assign_scaled_subtraction(self.context, 1. / self.probs.nrows, self.probs, self.true_labels)

    def fprop(self):
        self.x.softmax(self.context, self.probs)

    def add_callback(self, callback):
        self.context.add_callback(callback)

    def add_callback_on_loss(self, callback):
        self.true_labels_np = self.true_labels.to_host(self.context)
        self.probs_np = self.probs.to_host(self.context)
        self.add_callback(callback)

    def get_loss(self):
        if self.true_labels.dtype == 'int':
            idxs = range(self.probs_np.shape[0]), self.true_labels_np.flatten()
            return - np.mean(np.log(self.probs_np[idxs] + 1e-20))
        else:
            return - np.mean(np.log(np.sum(self.true_labels_np * self.probs_np, axis=1) + 1e-20))