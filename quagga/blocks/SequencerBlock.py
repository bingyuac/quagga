from itertools import izip
from quagga.context import Context
from quagga.matrix import MatrixList


class SequencerBlock(object):
    def __init__(self, block_class, params, sequences, output_names, mask=None, prev_names=None, paddings=None, reverse=False, device_id=None):
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
        if paddings:
            paddings = [e.register_usage(device_id) for e in paddings]
            self.prev_names = prev_names
            if reverse:
                self.temp_prev = []
                self.dL_dtemp_prev = []
        self._length = sequences[0].length
        self.blocks = []
        outputs = [[] for _ in output_names]
        for k in xrange(self._length):
            k = self._length.value - 1 - k if reverse else k
            args = params + sequences[k]
            if mask:
                args.append(self.mask[:, k])
            if prev_names:
                if k == 0:
                    prevs = paddings
                else:
                    prev_block = self.blocks[-1]
                    prevs = [getattr(prev_block, name) for name in prev_names]
                args += prevs
            args.append(device_id)
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
            if start_k:
                # connect paddings with current first block
                self.temp_prev = []
                self.dL_dtemp_prev = []
                for prev_name in self.prev_names:
                    self.temp_prev.append(getattr(self.blocks[start_k], prev_name))
                    prev = getattr(self.blocks[0], prev_name)
                    setattr(self.blocks[start_k], prev_name, prev)
                    if hasattr(self.blocks[0], 'dL_d' + prev_name):
                        prev_name = 'dL_d' + prev_name
                        self.dL_dtemp_prev.append(getattr(self.blocks[start_k], prev_name))
                        prev = getattr(self.blocks[0], prev_name)
                        setattr(self.blocks[start_k], prev_name, prev)
                    else:
                        self.dL_dtemp_prev.append(None)
            for k in xrange(start_k, max_input_sequence_len):
                self.blocks[k].fprop()
        else:
            for k in xrange(self._length):
                self.blocks[k].fprop()

    def bprop(self):
        if self.reverse:
            max_input_sequence_len = len(self.blocks)
            start_k = max_input_sequence_len - self._length.value
            for k in reversed(xrange(start_k, max_input_sequence_len)):
                self.blocks[k].bprop()
            if start_k:
                for i, prev_name in enumerate(self.prev_names):
                    prev = self.temp_prev[i]
                    setattr(self.blocks[start_k], prev_name, prev)
                    prev = self.dL_dtemp_prev[i]
                    if prev:
                        setattr(self.blocks[start_k], 'dL_d' + prev_name, prev)
                self.temp_prev = []
                self.dL_dtemp_prev = []
        else:
            for k in reversed(xrange(self._length)):
                self.blocks[k].bprop()