from itertools import izip
from quagga.context import Context
from quagga.matrix import MatrixList


class SequencerBlock(object):
    def __init__(self, block_class, params, sequences, output_names=None, prev_names=None, paddings=None, reverse=False, device_id=None, mask=None):
        """
        TODO

        :param block_class:
        :param params:
        :param sequences: sequences for izip-like iteration, must be MatrixList
        :param output_names: attribute names from block
        :param mask:
        :param prev_names: attributes from previous block that will be fed
                           into the current
        :param paddings:
        :param device_id:
        """

        self.context = Context(device_id)
        device_id = self.context.device_id
        self.reverse = reverse
        if mask:
            self.mask = mask.register_usage(device_id)
        self.prev_names = prev_names
        if prev_names and reverse:
            self.temp_prev = []
            self.dL_dtemp_prev = []
            self.k = None
        self._length = sequences[0].length
        self.blocks = []
        output_names = output_names if output_names else []
        outputs = [[] for _ in output_names]
        for k in xrange(self._length):
            k = self._length.value - 1 - k if reverse else k
            args = params + [s[k] for s in sequences]
            if prev_names:
                if k == (self._length - 1 if reverse else 0):
                    prevs = paddings
                else:
                    prev_block = self.blocks[-1]
                    prevs = [getattr(prev_block, name) for name in prev_names]
                args += prevs
            args.append(device_id)
            if mask:
                args.append(self.mask[:, k])
            self.blocks.append(block_class(*args))
            for i, output_name in enumerate(output_names):
                outputs[i].append(getattr(self.blocks[-1], output_name))
        for output_name, output in izip(output_names, outputs):
            output = output[::-1] if reverse else output
            output = MatrixList(output, self._length)
            setattr(self, output_name, output)

    def fprop(self):
        if self.reverse:
            max_input_sequence_len = len(self.blocks)
            start_k = max_input_sequence_len - self._length.value
            if start_k and self.prev_names:
                self.connect_block_with_padding(start_k)
            generator = xrange(start_k, max_input_sequence_len)
        else:
            generator = xrange(self._length)
        for k in generator:
            self.blocks[k].fprop()

    def bprop(self):
        if self.reverse:
            max_input_sequence_len = len(self.blocks)
            start_k = max_input_sequence_len - self._length.value
            generator = xrange(start_k, max_input_sequence_len)
        else:
            generator = xrange(self._length)
        # If there was no prev_names order is not important.
        # By not reversing it we can gain speed up.
        generator = reversed(generator) if self.prev_names else generator
        for k in generator:
            self.blocks[k].bprop()

    def connect_block_with_padding(self, k):
        self.disconnect_prev_first_block_with_padding()
        for name in self.prev_names:
            name = 'prev_' + name
            self.temp_prev.append(getattr(self.blocks[k], name))
            prev = getattr(self.blocks[0], name)
            setattr(self.blocks[k], name, prev)
            if hasattr(self.blocks[0], 'dL_d' + name):
                name = 'dL_d' + name
                self.dL_dtemp_prev.append(getattr(self.blocks[k], name))
                prev = getattr(self.blocks[0], name)
                setattr(self.blocks[k], name, prev)
            else:
                self.dL_dtemp_prev.append(None)
        self.k = k

    def disconnect_prev_first_block_with_padding(self):
        if self.temp_prev:
            for i, name in enumerate(self.prev_names):
                name = 'prev_' + name
                prev = self.temp_prev[i]
                setattr(self.blocks[self.k], name, prev)
                prev = self.dL_dtemp_prev[i]
                if prev:
                    setattr(self.blocks[self.k], 'dL_d' + name, prev)
            self.temp_prev = []
            self.dL_dtemp_prev = []