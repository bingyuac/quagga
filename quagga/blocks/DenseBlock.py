from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class DenseBlock(object):
    def __init__(self, init, features, activation_function, device_id=None):
        self.context = Context(device_id)
        self.w = Matrix.from_npa(init(), device_id=device_id)
        self.dL_dw = Matrix.empty_like(self.w, device_id)
        if features.bpropagable:
            self.features, self.dL_dfeatures = features.register_usage(self.context, self.context)
        else:
            self.features = features.register_usage(self.context)
        self.output = Connector(Matrix.empty(self.w.nrows, self.features.ncols, 'float', device_id), self.context, self.context)
        self._df_dpref = Matrix.empty(self.w.nrows, self.features.ncols, 'float', device_id)

        if activation_function == 'sigmoid':
            self.f = self.output.sigmoid
        elif activation_function == 'tanh':
            self.f = self.output.tanh
        elif activation_function == 'relu':
            self.f = self.output.relu
        self.back_prop = None

    @property
    def df_dpref(self):
        if self.back_prop:
            return self._df_dpref

    def set_training_mode(self):
        self.back_prop = True

    def set_testing_mode(self):
        self.back_prop = False

    def fprop(self):
        self._df_dpref.ncols = self.features.ncols
        self.output.ncols = self.features.ncols
        self.output.assign_dot(self.context, self.w, self.features)
        self.f(self.context, self.output, self.df_dpref)
        self.output.fprop()

    def bprop(self):
        dL_dpref = self.output.backward_matrix
        # dL/dpref = dL/df .* df/dpref
        dL_dpref.assign_hprod(self.context, dL_dpref, self.df_dpref)
        # dL/dw = dL/dpref * features.T
        self.dL_dw.assign_dot(self.context, dL_dpref, self.features, matrix_operation_b='T')
        # dL/dfeatures = w.T * dL/dpref
        if hasattr(self, 'dL_dfeatures'):
            self.dL_dfeatures.assign_dot(self.context, self.w, dL_dpref, matrix_operation_a='T')

    @property
    def params(self):
        return [(self.context, self.w)]

    @property
    def grads(self):
        return [(self.context, self.dL_dw)]