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