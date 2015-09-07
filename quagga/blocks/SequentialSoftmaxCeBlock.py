from quagga.matrix import Matrix
from quagga.context import Context
from quagga.matrix import MatrixList
from quagga.connector import Connector


class SequentialSoftmaxCeBlock(object):
    def __init__(self, x, true_labels, device_id=None):
        """
        TODO
        """
        if all(e.bpropagable for e in x):
            learning = True
        elif all(not e.bpropagable for e in x):
            learning = False
        else:
            raise ValueError('All elements of x should be bpropagable '
                             'or non-bpropagable. Mixed state is not allowed!')
        self.max_input_sequence_len = len(x)
        self._x = x
        if learning:
            self.x, self.dL_dx = zip(*[e.register_usage(self.context, self.context) for e in x])
        else:
            self.x = [e.register_usage(self.context) for e in x]

        if type(self.true_labels) is MatrixList:
            self.true_labels = [e.register_usage(self.context) for e in true_labels]
        else:
            self.true_labels = true_labels.register_usage(self.context)

        self.probs = Matrix.empty_like(x[0], self.context.device_id)
        self.output = Connector(self.output, self.context, self.context if learning else None)
        self.context = Context(device_id)

    def fprop(self):
        for i in xrange(len(self._x)):
            self.x[i].softmax(self.context, self.probs)

    def bprop(self):
        for e in self.a

    @property
    def loss(self):
        true_labels = self.true_labels.to_host()
        probs = self.probs.to_host()
        if self.true_labels.dtype == 'int':
            return - np.mean(np.log(probs[range(probs.shape[0]), true_labels.flatten()] + 1e-20))
        else:
            return - np.mean(np.sum(true_labels * np.log(probs + 1e-20), axis=1))