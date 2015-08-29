import os
import gzip
import cPickle
import numpy as np
from urllib import urlretrieve
from quagga import initializers
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector
from sklearn.preprocessing import OneHotEncoder
from quagga.blocks import DotBlock, NonlinearityBlock, DropoutBlock, SoftmaxCeBlock


def load_mnis_dataset():
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        urlretrieve('http://deeplearning.net/data/mnist/mnist.pkl.gz', filename)

    with gzip.open(filename, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
    return train_set[0], train_set[1], valid_set[0], valid_set[1], test_set[0], test_set[1]


class MnistMiniBatchesGenerator(object):
    def __init__(self, x, y, batch_size, randomize, infinity_generator, device_id):
        self.context = Context(device_id)
        device_id = self.context.device_id

        self.x = Matrix.from_npa(x.astype(np.float32), device_id=device_id)
        y = OneHotEncoder(dtype=np.float32, sparse=False).fit_transform(y[:, np.newaxis])
        self.y = Matrix.from_npa(y, device_id=device_id)
        self.batch_size = batch_size

        self.x_output = Matrix.empty(self.batch_size, self.x.ncols, device_id)
        self.x_output = Connector(self.x_output, self.context)
        self.y_output = Matrix.empty(self.batch_size, self.y.ncols, device_id)
        self.y_output = Connector(self.y_output, self.context)

        self.indices = np.arange(self.x.nrows, dtype=np.int32)
        self.q_indices = Matrix.empty(self.batch_size, 1, 'int32', device_id)
        self.randomize = randomize
        if self.randomize:
            self.rng = np.random.RandomState(42)
            self.rng.shuffle(self.indices)
        self.infinity_generator = infinity_generator
        self.i = 0

    def fprop(self):
        indices = self.indices[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        if len(indices) != self.batch_size:
            self.i = 0
            if not self.infinity_generator:
                raise StopIteration()
            elif self.randomize:
                self.rng.shuffle(self.indices)
                self.fprop()
        else:
            self.q_indices.to_device(self.context, indices)
            self.x.slice_rows(self.context, self.q_indices, self.x_output)
            self.y.slice_rows(self.context, self.q_indices, self.y_output)
            self.x_output.fprop()
            self.y_output.fprop()
            self.i += 1


if __name__=='__main__':
    train_x, train_y, _, _, _, _ = load_mnis_dataset()
    train_data_block = MnistMiniBatchesGenerator(x=train_x,
                                                 y=train_y,
                                                 batch_size=128,
                                                 randomize=True,
                                                 infinity_generator=True,
                                                 device_id=0)

    first_dot_block = DotBlock(W_init=initializers.Orthogonal(784, 500),
                               b_init=initializers.Constant(1, 500),
                               x=train_data_block.x_output,
                               device_id=0)
    first_nonl_block = NonlinearityBlock(x, nonlinearity, learning=True, device_id=None)
    first_dropout_block = DropoutBlock()
    second_dot_block = DotBlock(W_init, b_init, x, device_id=0)
    second_nonl_block = NonlinearityBlock()
    second_dropout_block = DropoutBlock()
    sce_dot_block = DotBlock(W_init, b_init, x, device_id=0)
    sce_block = = SoftmaxCeBlock()