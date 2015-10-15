from quagga.matrix import Matrix
from quagga.connector import Connector


class ParameterContainer(object):
    def __init__(self, **kwargs):
        self.parameters = {}
        self.trainable_parameters = {}
        for name, definition in kwargs.iteritems():
            device_id = definition['device_id']
            matrix = Matrix.from_npa(definition['init'](), device_id=device_id)
            if 'trainable' not in definition or definition['trainable']:
                param = Connector(matrix, device_id)
                self.trainable_parameters[name] = param
            else:
                param = Connector(matrix)
            self.parameters[name] = param

    def __getitem__(self, item):
        return self.parameters[item]

    def fprop(self):
        for param in self.parameters.itervalues():
            param.fprop()