# ----------------------------------------------------------------------------
# Copyright 2015 Grammarly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from itertools import izip

from quagga.utils import List
from quagga.context import Context


class SequencerBlock(object):
    """
    Parameters
    ----------
    block_class
    params
    sequences
    output_names
    prev_names
    paddings
    reverse
    device_id : int
        Defines the device's id on which the computation will take place


    Returns
    -------
    """
    def __init__(self, block_class, params, sequences, output_names=None, prev_names=None, paddings=None, reverse=False, device_id=None):
        context = Context(device_id)
        device_id = context.device_id
        self.reverse = reverse
        self.prev_names = prev_names
        if prev_names and reverse:
            self.temp_prev = []
            self.dL_dtemp_prev = []
            self.k = None
        self._length = sequences[0]._length
        self.blocks = []
        output_names = output_names if output_names else []
        outputs = [[] for _ in output_names]
        for k in xrange(self._length):
            k = self._length.value - 1 - k if reverse else k
            args = params + [s[k] for s in sequences]
            if prev_names:
                if k == (self._length.value - 1 if reverse else 0):
                    prevs = paddings
                else:
                    prev_block = self.blocks[-1]
                    prevs = [getattr(prev_block, name) for name in prev_names]
                args += prevs
            try:
                self.blocks.append(block_class(*args, device_id=device_id))
            except TypeError:
                self.blocks.append(block_class(*args))
            for i, output_name in enumerate(output_names):
                outputs[i].append(getattr(self.blocks[-1], output_name))
        for output_name, output in izip(output_names, outputs):
            output = output[::-1] if reverse else output
            output = List(output, self._length)
            setattr(self, output_name, output)

        if hasattr(self.blocks[0], 'calculate_loss') and hasattr(self.blocks[0], 'loss'):
            def calculate_loss(context):
                context.wait(*[self.blocks[i].context for i in xrange(self._length)])
                for i in xrange(self._length):
                    self.blocks[i].calculate_loss(context)
            self.calculate_loss = calculate_loss
            self.context = context
            SequencerBlock.loss = property(lambda self: [self.blocks[i].loss for i in xrange(self._length)])

    def fprop(self):
        if self.reverse:
            if self.prev_names:
                self.disconnect_prev_first_block_with_padding()
            max_input_sequence_len = len(self.blocks)
            start_k = max_input_sequence_len - self._length.value
            if 0 < start_k < max_input_sequence_len and self.prev_names:
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