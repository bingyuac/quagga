from quagga.matrix import Matrix
from quagga.matrix import MatrixList
from quagga.connector import Connector


class AddFirstBlock(object):
    def __init__(self, x):
        self.x = x
        zeros = Matrix.empty_like(x[0])
        zeros.sync_fill(0.0)
        zeros = Connector(zeros)
        self.output = MatrixList([zeros] + x[:])

    def fprop(self):
        self.output.set_length(len(self.x) + 1)

    def bprop(self):
        pass