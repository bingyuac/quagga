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
from quagga.matrix import ShapeElement


class List(object):
    def __init__(self, elements, length=None):
        self.elements = elements
        length = length if length is not None else len(elements)
        self._length = length if isinstance(length, ShapeElement) else ShapeElement(length)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length[:] = value

    def __getitem__(self, k):
        return self.elements[:self.length][k]

    def __iter__(self):
        return iter(self.elements[:self.length])

    def __len__(self):
        # TODO(sergii): fix everreting related to calling builtin len() function because it returns int instead of ShapeElement
        return self.length.value

    def __getattr__(self, name):
        attribute = getattr(self.elements[0], name)
        if hasattr(attribute, '__call__'):
            def method(*args, **kwargs):
                return [getattr(e, name)(*args, **kwargs) for e in self]
            setattr(self, name, method)
            return method
        else:
            raise AttributeError