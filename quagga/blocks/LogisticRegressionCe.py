import numpy as np
import ctypes as ct
from quagga.matrix import Matrix
from quagga.context import Context


class LogisticRegressionCe(object):
    """
    Logistic regression with cross entropy loss
    """


    def __init__(self, W_init, b_init, features, true_labels, learning=True, device_id=None):
        # TODO Rewrite this ass loss layer


        """
        TODO

        :param init: initializer for logistic regression weights
        :param features: connector that contains feature matrix.
        :param device_id:
        :param true_labels: connector that contains labels
        """
        if true_labels.nrows != features.nrows:
            raise ValueError('TODO!')

        self.context = Context(device_id)
        device_id = self.context.device_id
        self.W = Matrix.from_npa(W_init(), device_id=device_id)
        self.b = Matrix.from_npa(b_init(), device_id=device_id)
        if learning:
            self.dL_dW = Matrix.empty_like(self.W, device_id)
            self.dL_db = Matrix.empty_like(self.b, device_id)
            self.ones = Matrix.from_npa(np.ones((features.nrows, 1), np.float32), device_id=device_id)
        if learning and features.bpropagable:
            self.features, self.dL_dfeatures = features.register_usage(self.context, self.context)
        else:
            self.features = features.register_usage(self.context)
        self.true_labels = true_labels.register_usage(self.context)
        self.probs = Matrix.empty(true_labels.nrows, 1, 'float', device_id)

    def fprop(self):
        self.probs.assign_dot(self.context, self.features, self.W)
        self.probs.add(self.context, self.b)
        self.probs.sigmoid(self.context, self.probs)

    def bprop(self):
        # error = (probs - true_labels) / M
        self.probs.sub(self.context, self.true_labels)
        self.probs.scale(self.context, ct.c_float(1. / self.probs.nrows))
        # dL/dW = features.T * error
        self.dL_dW.assign_dot(self.context, self.features, self.probs, 'T')
        # dL/db = 1.T * error
        self.dL_db.assign_dot(self.context, self.ones, self.probs, 'T')
        # dL/dfeatures = error * w.T
        if hasattr(self, 'dL_dfeatures'):
            self.dL_dfeatures.assign_dot(self.context, self.probs, self.W, 'N', 'T')

    @property
    def loss(self):
        true_labels = self.true_labels.to_host()
        probs = self.probs.to_host()
        return - (true_labels * np.log(probs + 1e-20) +
                  (1.0 - true_labels) * np.log(1. - probs + 1e-20))

    @property
    def params(self):
        return [(self.context, self.W), (self.context, self.b)]

    @property
    def grads(self):
        return [(self.context, self.dL_dW), (self.context, self.dL_db)]