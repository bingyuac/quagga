import quagga.initializers
from quagga.matrix import Matrix
from quagga.connector import Connector


class ParameterContainer(object):
    def __init__(self, **kwargs):
        self.parameters = {}
        for name, definition in kwargs.iteritems():
            InitClass = getattr(quagga.initializers, definition['init'][0])
            init_args = definition['init'][1:]
            device_id = definition['device_id']
            matrix = Matrix.from_npa(InitClass(*init_args)(), device_id=device_id)
            param = Connector(matrix, device_id)
            self.parameters[name] = param
            setattr(self, name, param)
        self.npa_params = {}

    def fprop(self):
        for param in self.parameters.itervalues():
            param.fprop()

    def send_to_host(self, context, name):
        param = getattr(self, name)
        self.npa_params[name] = param.to_host(context)

    def get_npa_param(self, name):
        return self.npa_params[name]