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
import weakref
import operator
from numbers import Number


class ShapeElement(object):
    """
    Instances of this class are used in order to specify the shape of matrices.
    It is used for shape inference and shape propagation.

    Parameters
    ----------
    value : int
        Value of the ShapeElement instance
    """
    def __init__(self, value):
        self.value = value
        self.modif_handlers = set()

    def __setitem__(self, key, value):
        if key != slice(None):
            raise ValueError("key argument must be ':'")
        if isinstance(value, ShapeElement):
            value = weakref.proxy(value)
            self_proxy = weakref.proxy(self)
            modif_handler = lambda: self_proxy.__setitem__(slice(None),
                                                           value.value)
            value.add_modification_handler(modif_handler)
            self[:] = value.value
        elif isinstance(value, int):
            if self.value != value:
                self.value = value
                needless_handlers = []
                for modif_handler in self.modif_handlers:
                    try:
                        modif_handler()
                    except ReferenceError:
                        needless_handlers.append(modif_handler)
                self.modif_handlers.difference_update(needless_handlers)
        else:
            raise TypeError("'value' argument must be int or ShapeElement")

    def operation(self, other, op):
        """
        Performs corresponding operations on ``self`` and ``other``. Also,
        this method adds to ``self`` and ``other`` a corresponding modification
        handler that is used for shape propagation in case one of the elements
        (``self`` or ``other``) is changed.

        Parameters
        ----------
        other : ShapeElement or int
        op : operator (addition, multiplication, subtraction, etc.)

        Returns
        -------
        element : ShapeElement
            The returning instance contains the resulting value of the operation
            on ``other.value`` and ``self.value``.
        """
        if isinstance(other, ShapeElement):
            element = ShapeElement(op(self.value, other.value))
            element_proxy = weakref.proxy(element)
            self_proxy = weakref.proxy(self)
            other = weakref.proxy(other)
            handler = lambda: element_proxy.__setitem__(slice(None),
                                                        op(self_proxy.value,
                                                           other.value))
            other.add_modification_handler(handler)
        elif isinstance(other, int):
            element = ShapeElement(op(self.value, other))
            element_proxy = weakref.proxy(element)
            self_proxy = weakref.proxy(self)
            handler = lambda: element_proxy.__setitem__(slice(None),
                                                        op(self_proxy.value,
                                                           other))
        else:
            raise TypeError("'other' argument must be int or ShapeElement")
        self.add_modification_handler(handler)
        return element

    def __add__(self, other):
        return self.operation(other, operator.add)

    def __sub__(self, other):
        return self.operation(other, operator.sub)

    def __mul__(self, other):
        return self.operation(other, operator.mul)

    def __div__(self, other):
        return self.operation(other, operator.div)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.operation(other,
                              ShapeElement._get_reflected_op(operator.sub))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rdiv__(self, other):
        return self.operation(other,
                              ShapeElement._get_reflected_op(operator.div))

    def __eq__(self, other):
        if isinstance(other, Number):
            return self.value == other
        return self.value == other.value

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        if isinstance(other, Number):
            return self.value < other
        return self.value < other.value

    def __gt__(self, other):
        return not (self < other or self == other)

    def __le__(self, other):
        return not self > other

    def __ge__(self, other):
        return not self < other

    def __str__(self):
        return str(self.value)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __index__(self):
        if not isinstance(self.value, int):
            raise ValueError('Value of ShapeElement is not int!')
        return self.value

    def add_modification_handler(self, handler):
        """
        Adds a modification handler ``handler`` to a set of modification
        handlers of the instance. This function will be called in case of
        the instance's value is changed.

        Parameters
        ----------
        handler : python function

        """
        self.modif_handlers.add(handler)

    @staticmethod
    def _get_reflected_op(op):
        return lambda x, y: op(y, x)