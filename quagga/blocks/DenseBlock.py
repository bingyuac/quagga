from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class DenseBlock(object):
    def __init__(self, init, input, activation_function, device_id=None):
        self.context = Context()
        self.w = Matrix.from_npa(init(), device_id=device_id)
        if input._b_usage_context:
            self.input, self.dL_dinput = input.register_usage(self.context, self.context)
        else:
            self.input = input.register_usage(self.context)
        self.output = Connector(Matrix.empty(self.w.nrows, self.input.ncols, 'float', device_id), self.context, self.context)

    def fprop(self):
        self.output.ncols = self.input.ncols
        self.output.assign_dot(self.context, self.w, self.input)
        if activation_function == 'sigmoid':
            self.output.sigmoid(self.context, self.output)
        elif activation_function == 'tanh':
            self.output.tanh(self.context, self.output)
        elif activation_function == 'relu':
            self.output.relu(self.context, self.output)

    def bprop(self):
        pass