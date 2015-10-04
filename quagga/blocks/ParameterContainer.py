from quagga.matrix import Matrix
from quagga.connector import Connector


class ParameterContainer(object):
    def __init__(self, **kwargs):
        self.parameters = {}
        for name, definition in kwargs.iteritems():
            device_id = definition['device_id']
            matrix = Matrix.from_npa(definition['init'](), device_id=device_id)
            param = Connector(matrix, device_id)
            self.parameters[name] = param
        self.npa_params = {}

    def __getitem__(self, item):
        return self.parameters[item]

    def fprop(self):
        for param in self.parameters.itervalues():
            param.fprop()

    def send_to_host(self, context, name):
        self.npa_params[name] = self.parameters[name].to_host(context)

    def get_npa_param(self, name):
        return self.npa_params[name]