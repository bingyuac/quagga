from quagga.matrix import Matrix
from quagga.context import Context


class LogisticRegressionCe(object):
    """
    Logistic regression with cross entropy loss
    """

    def __init__(self, init, features, true_labels, device_id=None):
        """

        :param init: initializer for logistic regression weights
        :param features: connector that contains feature matrix.
        :param device_id:
        :param true_labels: connector that contains labels
        """
        if true_labels.ncols != features.ncols:
            raise ValueError('TODO!')

        self.w = Matrix.from_npa(init(), device_id=device_id)
        self.dL_dw = Matrix.empty_like(self.w, device_id)
        self.context = Context(device_id)
        if features.bpropagable:
            self.features, self.dL_dfeatures = features.register_usage(self.context, self.context)
        else:
            self.features = features.register_usage(self.context)
        self.true_labels = true_labels.register_usage(self.context)
        self.probs = Matrix.empty(true_labels.nrows, true_labels.ncols, 'float', device_id)
        self.ce = Matrix.empty(1, 1, 'float', device_id)

    # def get_block(self, features, true_labels):
    #     """
    #     Return the same block, which is only fprop capable
    #
    #
    #     :param features:
    #     :param true_labels:
    #     :return:
    #     """
    #     copy = object.__new__(LogisticRegressionCe)
    #     copy.w = self.w
    #     copy.context = self.context
    #     copy.features = features.register_usage(copy.context)
    #     copy.true_labels = true_labels.register_usage(copy.context)
    #     copy.probs = self.probs
    #     copy.ce = self.ce
    #     return copy

    def fprop(self):
        self.probs.ncols = self.features.ncols
        self.probs.assign_dot(self.context, self.w, self.features)
        self.probs.sigmoid(self.context, self.probs)

    def bprop(self):
        # error = probs - true_labels
        self.probs.sub(self.context, self.true_labels)
        # dL/dw = error * features.T
        self.dL_dw.assign_dot(self.context, self.probs, self.features, matrix_operation_b='T')
        # dL/dfeatures = w.T * error
        if hasattr(self, 'dL_dfeatures'):
            self.dL_dfeatures.assign_dot(self.context, self.w, self.probs, matrix_operation_a='T')

    @property
    def loss(self):
        self.ce.assign_cross_entropy(self.context, self.true_labels, self.probs)
        return self.ce.to_host()

    @property
    def params(self):
        return [(self.context, self.w)]

    @property
    def grads(self):
        return [(self.context, self.dL_dw)]