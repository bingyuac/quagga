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
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class GaussianNoiseBlock(object):
    """
    Adds Gaussian noise to the block's input. Adding Gaussian noise can be
    viewed as a regularization.


    Parameters
    ----------
    mean : float
            Expected value of Gaussian noise
    std : float
            Standard deviation of added Gaussian noise
    x : matrix
            Block's input
    seed : int
            Seed for :meth:`quagga.cuda.curand.random_generator()`
    device_id: int
            Defines the device's id on which the computation will take place
    """
    def __init__(self, mean, std, x, seed=42, device_id=None):
        self.mean = mean
        self.std = std
        self.f_context = Context(device_id)
        device_id = self.f_context.device_id
        self.generator = Matrix.get_random_generator(seed)
        if x.bpropagable:
            self.b_context = Context(device_id)
            self.x, self.dL_dx = x.register_usage(device_id, device_id)
        else:
            self.x = x.register_usage(device_id)
        self.output = Matrix.empty_like(self.x)
        self.output = Connector(self.output, device_id if x.bpropagable else None)
        self.training_mode = True

    def fprop(self):
        if self.training_mode:
            self.x.add_gaussian_noise(self.f_context, self.generator, self.mean, self.std, self.output)
        else:
            self.output.assign(self.f_context, self.x)
        self.output.fprop()

    def bprop(self):
        self.dL_dx.add(self.b_context, self.output.backward_matrix)

    def set_training_mode(self):
        self.training_mode = True

    def set_testing_mode(self):
        self.training_mode = False