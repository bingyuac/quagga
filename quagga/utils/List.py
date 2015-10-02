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
        return self.length

    def __getattr__(self, name):
        attribute = getattr(self.elements[0], name)
        if hasattr(attribute, '__call__'):
            def method(*args, **kwargs):
                return [getattr(e, name)(*args, **kwargs) for e in self]
            setattr(self, name, method)
            return method
        else:
            raise AttributeError