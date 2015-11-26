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
from quagga.connector import Connector


class ParameterContainer(object):
    def __init__(self, **kwargs):
        self.parameters = {}
        self.trainable_parameters = {}
        for name, definition in kwargs.iteritems():
            device_id = definition['device_id']
            matrix = Matrix.from_npa(definition['init'](), device_id=device_id)
            if 'trainable' not in definition or definition['trainable']:
                param = Connector(matrix, device_id)
                self.trainable_parameters[name] = param
            else:
                param = Connector(matrix)
            self.parameters[name] = param

    def __getitem__(self, item):
        return self.parameters[item]

    def fprop(self):
        for param in self.parameters.itervalues():
            param.fprop()