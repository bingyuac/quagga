from quagga.matrix import ShapeElement


class MatrixList(object):
    def __init__(self, matrices, length=None):
        self.matrices = matrices
        length = length if length is not None else len(matrices)
        self._length = length if isinstance(length, ShapeElement) else ShapeElement(length)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length[:] = value

    def __getitem__(self, k):
        if type(k) is slice:
            matrices = self.matrices[:self.length]
            return matrices[k]
        elif type(k) is int:
            if -self.length <= k < self.length:
                return self.matrices[k % self.length]
            else:
                raise IndexError('MatrixContainer index out of range')
        else:
            raise TypeError('MatrixList indices must be integer or slice')

    def __iter__(self):
        return iter(self.matrices[:self.length])

    def __len__(self):
        return self.length

    def to_host(self):
        return [e.to_host() for e in self]