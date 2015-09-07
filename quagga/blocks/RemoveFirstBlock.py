from quagga.matrix import MatrixList


class RemoveFirstBlock(object):
    def __init__(self, x):
        self.x = x
        # TODO add here first replacer zeros
        self.output = MatrixList(x.matrices[1:])

    def fprop(self):
        self.output.set_length(len(self.x) - 1)

    def bprop(self):
        pass