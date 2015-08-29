from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class NonlinearityBlock(object):
    def __init__(self, x, nonlinearity, device_id=None):
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.learning = x.bpropagable
        if self.learning:
            self._df_dpref = Matrix.empty_like(x, device_id)
            self.x, self.dL_dx = x.register_usage(self.context, self.context)
        else:
            self.x = x.register_usage(self.context)
        output = Matrix.empty_like(x, device_id)
        self.output = Connector(output, self.context, self.context if self.learning else None)
        if nonlinearity == 'sigmoid':
            self.f = self.x.sigmoid
        elif nonlinearity == 'tanh':
            self.f = self.x.tanh
        elif nonlinearity == 'relu':
            self.f = self.x.relu
        else:
            raise ValueError('TODO!')

    @property
    def df_dpref(self):
        if self.learning:
            return self._df_dpref

    def fprop(self):
        self.f(self.context, self.output, self.df_dpref)
        self.output.fprop()

    def bprop(self):
        if hasattr(self, 'dL_dx'):
            # dL/dpref = dL/df .* df/dpref
            dL_df = self.output.backward_matrix
            self.dL_dx.assign_hprod(self.context, dL_df, self.df_dpref)

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []