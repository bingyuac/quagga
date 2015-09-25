import copy
import quagga.blocks
import quagga.initializers
from collections import OrderedDict
from quagga.blocks import ParameterContainer


class Model(object):
    def __init__(self, model_definition, data_block):
        self.model_definition = model_definition
        self.data_block = data_block
        self.blocks = OrderedDict()
        self.blocks['data_block'] = data_block
        self.model_definition = copy.deepcopy(model_definition)

        for block_name, definition in model_definition.iteritems():
            kwargs = {}
            for key, value in definition.iteritems():
                if key == 'type':
                    BlockClass = getattr(quagga.blocks, value)
                elif isinstance(value, dict) and BlockClass is not ParameterContainer:
                    _block_name = value.keys()[0]
                    connector_name = value[_block_name]
                    kwargs[key] = getattr(self.blocks[_block_name], connector_name)
                else:
                    kwargs[key] = value
            self.blocks[block_name] = BlockClass(**kwargs)
        self.blocks = self.blocks.values()

        self.modeable_blocks = []
        self.loss_block = None
        for block in self.blocks:
            if hasattr(block, 'set_testing_mode') and \
                    hasattr(block, 'set_training_mode'):
                self.modeable_blocks.append(block)
            if hasattr(block, 'loss'):
                self.loss_block = block

        self.bpropable_blocks = []
        for block in reversed(self.blocks):
            if hasattr(block, 'bprop'):
                self.bpropable_blocks.append(block)

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