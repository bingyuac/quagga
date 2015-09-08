from quagga.matrix import MatrixList


class RemoveLastBlock(object):
    def __init__(self, x):
        self.x = x
        self.output = MatrixList(x[:])

    def fprop(self):
        self.output.set_length(len(self.x) - 1)

    def bprop(self):
        pass