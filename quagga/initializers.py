import numpy as np
from numbers import Number


rng = np.random.RandomState(seed=42)


class Constant(object):
    def __init__(self, nrows, ncols, val=0.0):
        self.shape = (nrows, ncols)
        self.nrows = nrows
        self.ncols = ncols
        self.val = val

    def __call__(self):
        c = np.empty(self.shape)
        c.fill(self.val)
        return c


class Orthogonal(object):
    def __init__(self, nrows, ncols):
        self.shape = (nrows, ncols)
        self.nrows = nrows
        self.ncols = ncols

    def __call__(self):
        a = rng.normal(0.0, 1.0, self.shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        return u if u.shape == self.shape else v


class Uniform(object):
    def __init__(self, nrows, ncols, init_range=None, activation_fun=None):
        self.shape = (nrows, ncols)
        self.nrows = nrows
        self.ncols = ncols
        self.init_range = init_range
        self.activation_function = activation_fun

    def __call__(self):
        if self.init_range is None:
            fan_in, fan_out = self.nrows, self.ncols
            if self.activation_function == 'tanh':
                bound = np.sqrt(6.0 / (fan_in + fan_out))
                init_range = (-bound, bound)
            elif self.activation_function == 'sigmoid':
                bound = 4 * np.sqrt(6.0 / (fan_in + fan_out))
                init_range = (-bound, bound)
            else:
                raise ValueError(self.activation_function)
        elif isinstance(self.init_range, Number):
            init_range = (-self.init_range, self.init_range)
        else:
            init_range = self.init_range
        return rng.uniform(low=init_range[0], high=init_range[1], size=self.shape)