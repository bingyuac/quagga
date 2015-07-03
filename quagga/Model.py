class Model(object):
    def __init__(self, *blocks):
        self.blocks = blocks
        self.bpropable_blocks = list(reversed([block for block in self.blocks if hasattr(block, 'bprop')]))
        self.modeable_blocks = list(reversed([block for block in self.blocks if hasattr(block, 'set_training_mode')]))
        if hasattr(self.blocks[-1], 'loss'):
            self.loss_block = self.blocks[-1]
        self.set_training_mode()

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
    def params(self):
        params = []
        for block in self.blocks:
            try:
                params.extend(block.params)
            except AttributeError:
                pass
        return params

    @property
    def grads(self):
        grads = []
        for block in self.blocks:
            try:
                grads.extend(block.grads)
            except AttributeError:
                pass
        return grads

    @property
    def loss(self):
        return self.loss_block.loss