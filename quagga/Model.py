class Model(object):
    def __init__(self, blocks):
        self.blocks = blocks
        self.loss_block = None
        self.modeable_blocks = []
        self.fpropable_blocks = []
        self.bpropable_blocks = []
        for block in self.blocks:
            if hasattr(block, 'calculate_loss') and hasattr(block, 'loss'):
                self.loss_block = block
            if hasattr(block, 'set_testing_mode') and \
                    hasattr(block, 'set_training_mode'):
                self.modeable_blocks.append(block)
            if hasattr(block, 'fprop'):
                self.fpropable_blocks.append(block)
            if hasattr(block, 'bprop'):
                self.bpropable_blocks.append(block)
        self.bpropable_blocks = list(reversed(self.bpropable_blocks))

    def set_training_mode(self):
        for block in self.modeable_blocks:
            block.set_training_mode()

    def set_testing_mode(self):
        for block in self.modeable_blocks:
            block.set_testing_mode()

    def fprop(self):
        for block in self.fpropable_blocks:
            block.fprop()

    def bprop(self):
        for block in self.bpropable_blocks:
            block.bprop()