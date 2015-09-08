import numpy as np
from itertools import izip
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.matrix import MatrixList
from quagga.connector import Connector
from quagga.blocks import SoftmaxCeBlock


class SequentialSoftmaxCeBlock(object):
    """
    TODO
    """

    def __init__(self, x_sequence, true_labels, device_id=None):
        if not all(e.bpropagable for e in x_sequence) or \
                not all(not e.bpropagable for e in x_sequence):
            raise ValueError('All elements of x should be bpropagable '
                             'or non-bpropagable. Mixed state is not allowed!')
        if type(true_labels) is not MatrixList:
            self.context = Context(device_id)
            true_labels = true_labels.register_usage(self.context)
            true_labels = [true_labels[:, i] for i in xrange(true_labels.ncols)]
        self.softmax_ce_blocks = [SoftmaxCeBlock(x, e, device_id) for x, e in izip(x_sequence, true_labels)]
        self.x_sequence = x_sequence
        self.max_input_sequence_len = len(x_sequence)

    def fprop(self):
        n = len(self.x_sequence)
        if n > self.max_input_sequence_len:
            raise ValueError('Sequence has length: {} that is too long. '
                             'The maximum is: {}'.
                             format(n, self.max_input_sequence_len))
        if hasattr(self, 'context'):
            self.context.block(e.context for e in self.softmax_ce_blocks[:n])
        for e in self.softmax_ce_blocks[:n]:
            e.fprop()

    def bprop(self):
        for e in self.a:
            pass

    @property
    def loss(self):
        n = len(self.x_sequence)
        return [block.loss for block in self.softmax_ce_blocks[:n]]


class _SoftmaxCeBlock(object):
    def __init__(self, x, true_labels, device_id):
        if x.nrows != true_labels.nrows:
            raise ValueError('TODO!')
        self.context = Context(device_id)
        if x.bpropagable:
            self.x, self.dL_dx = x.register_usage(self.context, self.context)
        else:
            self.x = x.register_usage(self.context)
        self.probs = Matrix.empty_like(self.x, device_id=self.context.device_id)
        if isinstance(true_labels, Connector):
            self.true_labels = true_labels.register_usage(self.context)
        else:
            self.true_labels = true_labels
        # error = (probs - true_labels) / M
        if self.true_labels.dtype == 'int':
            self.bprop = lambda self: self.dL_dx.assign_softmax_ce_derivative(self.context, self.probs, self.true_labels)
        else:
            self.bprop = lambda self: self.dL_dx.assign_scaled_subtraction(self.context, 1. / self.probs.nrows, self.probs, self.true_labels)

    def fprop(self):
        self.x.softmax(self.context, self.probs)

    @property
    def loss(self):
        true_labels = self.true_labels.to_host()
        probs = self.probs.to_host()
        if self.true_labels.dtype == 'int':
            idxs = range(probs.shape[0]), true_labels.flatten()
            return - np.mean(np.log(probs[idxs] + 1e-20))
        else:
            return - np.mean(np.log(np.sum(true_labels * probs, axis=1) + 1e-20))