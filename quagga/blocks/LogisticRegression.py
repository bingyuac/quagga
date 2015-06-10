from quagga.matrix import Matrix


class LogisticRegression(object):
    """
    Logistic regression with cross entropy loss
    """

    def __init__(self, logistic_init, features, true_label):
        self.w = Matrix.from_npa(logistic_init())
        self.features = features
        self.true_label = true_label

    def fprop(self):
        pass

    def bprop(self):
        pass