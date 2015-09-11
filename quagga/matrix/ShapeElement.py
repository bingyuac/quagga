import operator


class ShapeElement(object):
    def __init__(self, value):
        self.value = value
        self.modif_handlers = []

    def __setitem__(self, key, value):
        if key != slice(None):
            raise ValueError("key argument must be ':'")
        if isinstance(value, ShapeElement):
            modif_handler = lambda: self.__setitem__(slice(None), value.value)
            value.add_modification_handler(modif_handler)
            self[:] = value.value
        elif isinstance(value, int):
            if self.value != value:
                self.value = value
                for modif_handler in self.modif_handlers:
                    modif_handler()
        else:
            raise TypeError("'value' argument must be int")

    def operation(self, other, op):
        if isinstance(other, ShapeElement):
            element = ShapeElement(op(self.value, other.value))
            modif_handler = lambda: element.__setitem__(slice(None), op(self.value, other.value))
            other.add_modification_handler(modif_handler)
        elif isinstance(other, int):
            element = ShapeElement(op(self.value, other))
            modif_handler = lambda: element.__setitem__(slice(None), op(self.value, other))
        else:
            raise TypeError
        self.add_modification_handler(modif_handler)
        return element

    def __add__(self, other):
        return self.operation(other, operator.add)

    def __sub__(self, other):
        return self.operation(other, operator.sub)

    def __mul__(self, other):
        return self.operation(other, operator.mul)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def add_modification_handler(self, fun):
        self.modif_handlers.append(fun)


a = ShapeElement(10)
b = ShapeElement(10)
f = ShapeElement(None)
c = a + b
d = a + 1 + c
e = 4 * d + a
f[:] = e
print a.value, b.value, c.value, d.value, e.value, f.value
a[:] = 23
print a.value, b.value, c.value, d.value, e.value, f.value