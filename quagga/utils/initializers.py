import h5py
import numpy as np
from numbers import Number


rng = np.random.RandomState(seed=42)


class Constant(object):
    def __init__(self, nrows, ncols, val=0.0):
        self.shape = (nrows, ncols)
        self.val = val

    def __call__(self):
        c = np.empty(self.shape, dtype=np.float32, order='F')
        c.fill(self.val)
        return c


class Orthogonal(object):
    def __init__(self, nrows, ncols):
        self.shape = (nrows, ncols)

    def __call__(self):
        a = rng.normal(0.0, 1.0, self.shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        a = u if u.shape == self.shape else v
        return np.asfortranarray(a, np.float32)


class Xavier(object):
    def __init__(self, nrows, ncols):
        self.shape = (nrows, ncols)

    def __call__(self):
        amp = np.sqrt(6.0 / (self.shape[0] + self.shape[1]))
        a = rng.uniform(-amp, amp, self.shape)
        return np.asfortranarray(a, np.float32)


class Uniform(object):
    def __init__(self, nrows, ncols, init_range=None):
        self.shape = (nrows, ncols)
        self.init_range = init_range

    def __call__(self):
        if self.init_range is None:
            fan_in, fan_out = self.shape
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            init_range = (-bound, bound)
        elif isinstance(self.init_range, Number):
            init_range = (-self.init_range, self.init_range)
        else:
            init_range = self.init_range
        a = rng.uniform(low=init_range[0], high=init_range[1], size=self.shape)
        return np.asfortranarray(a, np.float32)


class H5pyInitializer(object):
    def __init__(self, path, key):
        with h5py.File(path, 'r') as f:
            matrix = f[key][...]
        self.matrix = matrix.astype(np.float32)

    def __call__(self):
        return np.copy(self.matrix)