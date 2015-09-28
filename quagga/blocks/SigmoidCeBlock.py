import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class SigmoidCeBlock(object):
    """
    Sigmoid nonlinearity with mean cross entropy loss
    """

    def __init__(self, x, true_labels, device_id=None, mask=None):
        self.context = Context(device_id)
        device_id = self.context.device_id
        if x.bpropagable:
            self.x, self.dL_dx = x.register_usage(device_id, device_id)
        else:
            self.x = x.register_usage(device_id)
        self.true_labels = true_labels.register_usage(device_id)
        self.probs = Matrix.empty_like(self.x)
        if isinstance(mask, Connector):
            self.mask = mask.register_usage(device_id)
        else:
            self.mask = mask
        self.loss = None
        self._calculate_ce_loss = Context.callback(self._calculate_ce_loss)

    def fprop(self):
        self.x.sigmoid(self.context, self.probs)

    def bprop(self):
        # error = (probs - true_labels) / M
        self.dL_dx.add_scaled_subtraction(self.context, 1. / self.probs.nrows, self.probs, self.true_labels)
        if self.mask:
            self.dL_dx.hprod(self.context, self.mask)

    def calculate_loss(self, context):
        true_labels_np = self.true_labels.to_host(context)
        probs_np = self.probs.to_host(context)
        if self.mask:
            mask = self.mask.to_host(context)
            context.add_callback(self._calculate_ce_loss, true_labels_np, probs_np, mask)
        else:
            context.add_callback(self._calculate_ce_loss, true_labels_np, probs_np)

    def _calculate_ce_loss(self, true_labels_np, probs_np, mask=None):
        logs = true_labels_np * np.log(probs_np + 1e-20) + \
               (1.0 - true_labels_np) * np.log(1. - probs_np + 1e-20)
        if mask:
            logs *= mask[:, 0]
        self.loss = - np.mean(logs)