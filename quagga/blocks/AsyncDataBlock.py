from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class AsyncDataBlock(object):
    def __init__(self, data_source, max_buffer_size):
        self.max_buffer_size = max_buffer_size
        self.data_source = data_source
        self.device_buffers = [Matrix.empty(max_buffer_size, 1, 'int'),
                               Matrix.empty(max_buffer_size, 1, 'int')]
        self.contexts = [Context(), Context()]
        self.data = Connector(None)
        self.i = 0
        self.allocate_data()

    def fprop(self):
        self.data.forward_matrix = self.device_buffers[self.i]
        self.data.forward_context = self.contexts[self.i]
        self.i = (self.i + 1) % 2
        self.allocate_data()

    def allocate_data(self):
        data = self.data_source.get()
        if len(data) > self.max_buffer_size:
            raise ValueError('Datum is too big! Max len is {}'.
                             format(self.max_buffer_size))
        self.device_buffers[self.i].to_device(self.contexts[self.i], data)