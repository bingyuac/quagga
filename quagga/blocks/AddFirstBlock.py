from quagga.matrix import MatrixList


class AddFirstBlock(object):
    def __init__(self, x):
        self.x = x
        # TODO add here last replacer zeros
        self.output = MatrixList(x.matrices)

    def fprop(self):
        self.output.set_length(len(self.x) - 1)

    def bprop(self):
        pass