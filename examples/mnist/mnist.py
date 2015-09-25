import os
import gzip
import json
import cPickle
import logging
import numpy as np
from quagga import Model
from quagga.cuda import cudart
from urllib import urlretrieve
from quagga.matrix import Matrix
from quagga.context import Context
from collections import OrderedDict
from quagga.connector import Connector
from quagga.optimizers import Optimizer
from quagga.optimizers.steps import SgdStep
from quagga.blocks import ParameterContainer
from quagga.optimizers.observers import Saver
from quagga.optimizers.observers import ValidLossTracker
from quagga.optimizers.observers import TrainLossTracker
from quagga.optimizers.policies import FixedLearningRatePolicy
from quagga.optimizers.stopping_criteria import MaxIterCriterion


def get_logger(file_name):
    logger = logging.getLogger('train_logger')
    handler = logging.FileHandler(file_name, mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%d-%m-%Y %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_mnis_dataset():
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
    with open('mnist.json') as f:
        model_definition = json.load(f, object_pairs_hook=OrderedDict)

    train_x, train_y, valid_x, valid_y, _, _ = load_mnis_dataset()
    data_block = MnistMiniBatchesGenerator(train_x, train_y, valid_x, valid_y, batch_size=1024, device_id=0)
    model = Model(model_definition, data_block)
    logger = get_logger('train.log')
    learning_rate_policy = FixedLearningRatePolicy(0.01)
    train_loss_tracker = TrainLossTracker(model, 200, logger)
    valid_loss_tracker = ValidLossTracker(model, 200, logger)
    saver = Saver(model, 5000, 'mnist_trained.json', 'mnist_parameters.hdf5', logger)

    trainable_params = []
    for block in model.blocks:
        if isinstance(block, ParameterContainer):
            for param in block.parameters.itervalues():
                trainable_params.append(param)
    sgd_step = SgdStep(trainable_params, learning_rate_policy)
    data_block.blocking_contexts = sgd_step.blocking_contexts
    criterion = MaxIterCriterion(20000)

    optimizer = Optimizer(criterion, model)
    optimizer.add_observer(sgd_step)
    optimizer.add_observer(train_loss_tracker)
    optimizer.add_observer(valid_loss_tracker)
    optimizer.add_observer(saver)
    optimizer.add_observer(criterion)
    optimizer.optimize()

    for device_id in xrange(cudart.cuda_get_device_count()):
        cudart.cuda_set_device(device_id)
        cudart.cuda_device_synchronize()