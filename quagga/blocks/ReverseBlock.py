from quagga.blocks import Connector


class ReverseBlock(object):
    def __init__(self, input):
        self.input = input
        self.output = Connector(None, None)

    def fprop(self):
        pass

    def bprop(self):
        pass