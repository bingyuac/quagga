import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context


class SigmoidCeBlock(object):
    """
    Sigmoid nonlinearity with mean cross entropy loss
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
        self.loss = None
        self._calculate_ce_loss = Context.callback(self._calculate_ce_loss)

    def fprop(self):
        self.x.sigmoid(self.context, self.probs)

    def bprop(self):
        # error = (probs - true_labels) / M
        self.dL_dx.add_scaled_subtraction(self.context, 1. / self.probs.nrows, self.probs, self.true_labels)

    def calculate_loss(self, context):
        true_labels_np = self.true_labels.to_host(context)
        probs_np = self.probs.to_host(context)
        context.add_callback(self._calculate_ce_loss, true_labels_np, probs_np)

    def _calculate_ce_loss(self, true_labels_np, probs_np):
        self.loss = - (true_labels_np * np.log(probs_np + 1e-20) +
                      (1.0 - true_labels_np) * np.log(1. - probs_np + 1e-20))