from quagga.matrix import MatrixClass


class LogisticRegression(object):
    def __init__(self, p_type, logistic_init):
        self.W_hy = MatrixClass[p_type].from_npa(logistic_init())

    def forward_propagation(self, features):
        pass

    def backward_propagation(self, features, true_label):
        predicted_prob = self.forward_propagation()