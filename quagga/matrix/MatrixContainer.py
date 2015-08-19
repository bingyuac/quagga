class MatrixContainer(object):
    def __init__(self, matrices):
        self.matrices = matrices
        self.length = len(matrices)

    def set_length(self, length):
        self.length = length

    def __getitem__(self, k):
        if -self.length <= k < self.length:
            return self.matrices[k % self.length]
        else:
            raise IndexError('MatrixContainer index out of range')

    def __len__(self):
        return self.length