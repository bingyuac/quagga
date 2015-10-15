import os
import gzip
import cPickle
import logging
import numpy as np
from quagga import Model
from quagga.cuda import cudart
from urllib import urlretrieve
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import DotBlock
from quagga.blocks import DropoutBlock
from quagga.connector import Connector
from quagga.optimizers import Optimizer
from quagga.blocks import SoftmaxCeBlock
from quagga.optimizers.steps import NagStep
from quagga.blocks import NonlinearityBlock
from quagga.blocks import ParameterContainer
from quagga.utils.initializers import Constant
from quagga.utils.initializers import Orthogonal
from quagga.optimizers.observers import Hdf5Saver
from quagga.optimizers.observers import ValidLossTracker
from quagga.optimizers.observers import TrainLossTracker
from quagga.optimizers.policies import FixedMomentumPolicy
from quagga.optimizers.policies import FixedLearningRatePolicy
from quagga.optimizers.stopping_criteria import MaxIterCriterion


def get_logger(file_name):
    logger = logging.getLogger('train_logger')
    handler = logging.FileHandler(file_name, mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%d-%m-%Y %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_mnist_dataset():
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        urlretrieve('http://deeplearning.net/data/mnist/mnist.pkl.gz', filename)
    with gzip.open(filename, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
    return train_set[0], train_set[1], valid_set[0], valid_set[1], test_set[0], test_set[1]


class MnistMiniBatchesGenerator(object):
    def __init__(self, train_x, train_y, valid_x, valid_y, batch_size, device_id):
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.train_x = Matrix.from_npa(train_x.T.astype(np.float32), device_id=device_id)
        self.valid_x = Matrix.from_npa(valid_x.T.astype(np.float32), device_id=device_id)
        self.train_y = Matrix.from_npa(train_y[:, np.newaxis], 'int', device_id=device_id)
        self.valid_y = Matrix.from_npa(valid_y[:, np.newaxis], 'int', device_id=device_id)
        self.batch_size = batch_size

        x = Matrix.empty(self.batch_size, self.train_x.nrows, device_id=device_id)
        y = Matrix.empty(self.batch_size, 1, 'int', device_id)
        self.x = Connector(x)
        self.y = Connector(y)

        self.train_indices = np.arange(int(self.train_x.ncols), dtype=np.int32)
        self.valid_indices = np.arange(int(self.valid_x.ncols), dtype=np.int32)
        self.indices = Matrix.empty(self.batch_size, 1, 'int', device_id)
        self.rng = np.random.RandomState(42)
        self.rng.shuffle(self.train_indices)
        self.train_i = 0
        self.valid_i = 0
        self.training_mode = True

        self.blocking_contexts = None

    def set_training_mode(self):
        self.training_mode = True

    def set_testing_mode(self):
        self.training_mode = False

    def fprop(self):
        indices = self.train_indices if self.training_mode else self.valid_indices
        i = self.train_i if self.training_mode else self.valid_i
        x = self.train_x if self.training_mode else self.valid_x
        y = self.train_y if self.training_mode else self.valid_y

        indices = indices[self.batch_size * i:self.batch_size * (i + 1)]
        indices = np.asfortranarray(indices[:, np.newaxis])

        if self.training_mode:
            self.train_i += 1
        else:
            self.valid_i += 1

        if indices.size:
            self.indices.assign_npa(self.context, indices)
            self.x.nrows = indices.size
            self.y.nrows = indices.size
            self.context.wait(*self.blocking_contexts)
            x.slice_columns_and_transpose(self.context, self.indices, self.x)
            y.slice_rows(self.context, self.indices, self.y)
            self.x.fprop()
            self.y.fprop()
        else:
            if self.training_mode:
                self.train_i = 0
                self.rng.shuffle(self.train_indices)
                self.fprop()
            else:
                self.valid_i = 0
                raise StopIteration()


if __name__ == '__main__':
    train_x, train_y, valid_x, valid_y, _, _ = load_mnist_dataset()
    p = ParameterContainer(first_dot_block_W={'init': Orthogonal(784, 1024),
                                              'device_id': 0},
                           first_dot_block_b={'init': Constant(1, 1024),
                                              'device_id': 0},
                           second_dot_block_W={'init': Orthogonal(1024, 512),
                                               'device_id': 0},
                           second_dot_block_b={'init': Constant(1, 512),
                                               'device_id': 0},
                           sce_dot_block_W={'init': Orthogonal(512, 10),
                                            'device_id': 0},
                           sce_dot_block_b={'init': Constant(1, 10),
                                            'device_id': 0})
    data_block = MnistMiniBatchesGenerator(train_x, train_y, valid_x, valid_y,
                                           batch_size=1024, device_id=0)
    input_data_dropout_block = DropoutBlock(x=data_block.x,
                                            dropout_prob=0.2,
                                            device_id=0)
    first_dot_block = DotBlock(W=p['first_dot_block_W'],
                               b=p['first_dot_block_b'],
                               x=input_data_dropout_block.output,
                               device_id=0)
    first_nonl_block = NonlinearityBlock(x=first_dot_block.output,
                                         nonlinearity='relu',
                                         device_id=0)
    first_dropout_block = DropoutBlock(x=first_nonl_block.output,
                                       dropout_prob=0.5,
                                       device_id=0)
    second_dot_block = DotBlock(W=p['second_dot_block_W'],
                                b=p['second_dot_block_b'],
                                x=first_dropout_block.output,
                                device_id=0)
    second_nonl_block = NonlinearityBlock(x=second_dot_block.output,
                                          nonlinearity='relu',
                                          device_id=0)
    second_dropout_block = DropoutBlock(x=second_nonl_block.output,
                                        dropout_prob=0.5,
                                        device_id=0)
    sce_dot_block = DotBlock(W=p['sce_dot_block_W'],
                             b=p['sce_dot_block_b'],
                             x=second_dropout_block.output,
                             device_id=0)
    sce_block = SoftmaxCeBlock(x=sce_dot_block.output,
                               true_labels=data_block.y,
                               device_id=0)
    model = Model([p, data_block, input_data_dropout_block,
                   first_dot_block, first_nonl_block, first_dropout_block,
                   second_dot_block, second_nonl_block, second_dropout_block,
                   sce_dot_block, sce_block])

    logger = get_logger('train.log')
    learning_rate_policy = FixedLearningRatePolicy(0.01)
    momentum_policy = FixedMomentumPolicy(0.95)
    train_loss_tracker = TrainLossTracker(model, 200, logger)
    valid_loss_tracker = ValidLossTracker(model, 200, logger)
    saver = Hdf5Saver(p.parameters, 5000, 'mnist_parameters.hdf5', logger)
    nag_step = NagStep(p.parameters.values(), learning_rate_policy, momentum_policy)
    data_block.blocking_contexts = nag_step.blocking_contexts
    criterion = MaxIterCriterion(20000)

    optimizer = Optimizer(criterion, model)
    optimizer.add_observer(nag_step)
    optimizer.add_observer(train_loss_tracker)
    optimizer.add_observer(valid_loss_tracker)
    optimizer.add_observer(saver)
    optimizer.add_observer(criterion)
    optimizer.optimize()

    for device_id in xrange(cudart.cuda_get_device_count()):
        cudart.cuda_set_device(device_id)
        cudart.cuda_device_synchronize()