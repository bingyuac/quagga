import h5py
import quagga.blocks
import quagga.initializers
from collections import OrderedDict


class Model(object):
    def __init__(self, model_definition, data_block):
        self.model_definition = model_definition
        self.data_block = data_block
        self.blocks = OrderedDict()
        self.blocks['data_block'] = data_block
        h5_file = None
        for block_name, definition in model_definition.iteritems():
            kwargs = {}
            for key, value in definition.iteritems():
                if key == 'type':
                    BlockClass = getattr(quagga.blocks, value)
                elif isinstance(value, list):
                    if len(value) == 1:
                        if not h5_file:
                            h5_file = h5py.File(value[0], 'r')
                        temp = h5_file[key][...]
                        kwargs[key] = lambda: temp
                        kwargs[key].nrows = temp.shape[0]
                        kwargs[key].ncols = temp.shape[1]
                    else:
                        InitializerClass = getattr(quagga.initializers, value[0])
                        kwargs[key] = InitializerClass(*value[1:])
                elif isinstance(value, dict):
                    kwargs[key] = getattr(self.blocks[value.keys()[0]], value.values()[0])
                else:
                    kwargs[key] = value
            self.blocks[block_name] = BlockClass(**kwargs)
        if h5_file:
            h5_file.close()
        self.blocks_names = self.blocks.keys()
        self.blocks = self.blocks.values()

        self.params = []
        self.grads = []
        for block in self.blocks:
            try:
                self.params.extend(block.params)
                self.grads.extend(block.grads)
            except AttributeError:
                pass

        self.modeable_blocks = []
        for block in self.blocks:
            try:
                block.set_testing_mode()
                block.set_training_mode()
                self.modeable_blocks.append(block)
            except AttributeError:
                pass

        self.loss_block = self.blocks[-1]
        self.bpropable_blocks = list(reversed(self.blocks[1:]))

    def set_training_mode(self):
        for block in self.modeable_blocks:
            block.set_training_mode()

    def set_testing_mode(self):
        for block in self.modeable_blocks:
            block.set_testing_mode()

    def fprop(self):
        for block in self.blocks:
            block.fprop()

    def bprop(self):
        for block in self.bpropable_blocks:
            block.bprop()

    @property
    def loss(self):
        return self.loss_block.loss