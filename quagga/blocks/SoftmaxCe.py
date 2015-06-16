from quagga.matrix import Matrix
from quagga.context import Context


class SoftmaxCe(object):
    """
    Softmax with cross entropy loss
    """

    def __init__(self, init, features, true_labels, max_instance_num, propagate_error=True):
        """

        :param init: initializer for softmax weights
        :param features: connector that contains feature matrix.
        :param true_labels: connector that contains labels
        :param max_instance_num:
        :param propagate_error:
        """
        self.w = Matrix.from_npa(init())
        self.dL_dw = Matrix.empty_like(self.w)
        self.features = features
        self.true_labels = true_labels
        self.probs = Matrix.empty(true_labels.nrows, max_instance_num, 'float')
        self.context = Context()
        if propagate_error:
            self.dL_dfeatures = Matrix.empty(features.nrows, max_instance_num, 'float')
            self.features.register_user(self, self.context, self.dL_dfeatures)
        self.propagate_error = propagate_error
        self.max_instance_num = max_instance_num

    def fprop(self):
        n = self.features.ncols
        if n > self.max_instance_num:
            raise ValueError('There is {} instances, that is too big. '
                             'The maximum is: {}'.
                             format(n, self.max_instance_num))
        self.probs.ncols = n
        self.dL_dfeatures.ncols = n
        self.features.forward_block(self.context)
        self.probs.assign_dot(self.context, self.w, self.features)
        self.probs.softmax(self.context, self.probs)

    def bprop(self):
        # error = probs - true_labels
        self.true_labels.forward_block(self.context)
        self.probs.sub(self.context, self.true_labels)
        # dL/dw = error * features.T
        self.dL_dw.assign_dot(self.context, self.probs, self.features, matrix_operation_b='T')
        # dL/dfeatures = w.T * error
        if self.propagate_error:
            self.dL_dfeatures.assign_dot(self.context, self.w, self.probs, matrix_operation_a='T')

    @property
    def parameters(self):
        return self.w

    @property
    def d(self):
        return self.dL_dw