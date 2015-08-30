import os
import gzip
import json
import cPickle
import logging
import numpy as np
from quagga import Model
from urllib import urlretrieve
from quagga.matrix import Matrix
from quagga.context import Context
from collections import OrderedDict
from quagga.connector import Connector
from quagga.optimizers import SgdOptimizer
from sklearn.preprocessing import OneHotEncoder
from quagga.optimizers.observers import ValidLossTracker
from quagga.optimizers.observers import TrainLossTracker
from quagga.optimizers.policies import FixedLearningRatePolicy


def get_logger():
    logger = logging.getLogger('train_logger')
    handler = logging.FileHandler('train.log', mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%d-%m-%Y %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_mnis_dataset():
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print 'dow'
        urlretrieve('http://deeplearning.net/data/mnist/mnist.pkl.gz', filename)

    with gzip.open(filename, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
    return train_set[0], train_set[1], valid_set[0], valid_set[1], test_set[0], test_set[1]


class MnistMiniBatchesGenerator(object):
    def __init__(self, train_x, train_y, valid_x, valid_y, batch_size, device_id):
        self.blocking_context = None
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.train_x = Matrix.from_npa(train_x.T.astype(np.float32), device_id=device_id)
        self.valid_x = Matrix.from_npa(valid_x.T.astype(np.float32), device_id=device_id)
        one_hot_encoder = OneHotEncoder(dtype=np.float32, sparse=False)
        one_hot_encoder.fit(train_y[:, np.newaxis])
        train_y = one_hot_encoder.transform(train_y[:, np.newaxis])
        valid_y = one_hot_encoder.transform(valid_y[:, np.newaxis])
        self.train_y = Matrix.from_npa(train_y.T, device_id=device_id)
        self.valid_y = Matrix.from_npa(valid_y.T, device_id=device_id)
        self.batch_size = batch_size

        self.x_output = Matrix.empty(self.batch_size, self.train_x.nrows, device_id=device_id)
        self.x_output = Connector(self.x_output, self.context)
        self.y_output = Matrix.empty(self.batch_size, self.train_y.nrows, device_id=device_id)
        self.y_output = Connector(self.y_output, self.context)

        self.train_indices = np.arange(self.train_x.ncols, dtype=np.int32)
        self.valid_indices = np.arange(self.valid_x.ncols, dtype=np.int32)
        self.q_indices = Matrix.empty(1, self.batch_size, 'int', device_id=device_id)
        self.rng = np.random.RandomState(42)
        self.rng.shuffle(self.train_indices)
        self.train_i = 0
        self.valid_i = 0
        self.training_mode = True

    def set_training_mode(self):
        self.training_mode = True

    def set_testing_mode(self):
        self.training_mode = False

    def fprop(self):
        if self.training_mode:
            indices = self.train_indices[self.batch_size * self.train_i:
                                         self.batch_size * (self.train_i + 1)]
            indices = np.asfortranarray(indices[:, np.newaxis])
            self.train_i += 1
            if len(indices) == self.batch_size:
                self.q_indices.to_device(self.context, indices)
                self.context.wait(self.blocking_context)
                self.train_x.slice_columns_and_transpose(self.context, self.q_indices, self.x_output)
                self.train_y.slice_columns_and_transpose(self.context, self.q_indices, self.y_output)
                self.x_output.fprop()
                self.y_output.fprop()
            else:
                self.train_i = 0
                self.rng.shuffle(self.train_indices)
                self.fprop()
        else:
            indices = self.valid_indices[self.batch_size * self.valid_i:
                                         self.batch_size * (self.valid_i + 1)]
            indices = np.asfortranarray(indices[:, np.newaxis])
            self.valid_i += 1
            if len(indices) == self.batch_size:
                self.q_indices.to_device(self.context, indices)
                self.context.wait(self.blocking_context)
                self.valid_x.slice_columns_and_transpose(self.context, self.q_indices, self.x_output)
                self.valid_y.slice_columns_and_transpose(self.context, self.q_indices, self.y_output)
                self.x_output.fprop()
                self.y_output.fprop()
            else:
                self.valid_i = 0
                raise StopIteration()


if __name__ == '__main__':
    train_x, train_y, valid_x, valid_y, _, _ = load_mnis_dataset()
    data_block = MnistMiniBatchesGenerator(train_x, train_y, valid_x, valid_y, batch_size=128, device_id=0)
    with open('mnist.json') as f:
        model_definition = json.load(f, object_pairs_hook=OrderedDict)
    model = Model(model_definition, data_block)
    logger = get_logger()
    learning_rate_policy = FixedLearningRatePolicy(0.01)
    train_loss_tracker = TrainLossTracker(model, 2000, logger)
    valid_loss_tracker = ValidLossTracker(model, 2000, logger)
    sgd_optimizer = SgdOptimizer(50000, learning_rate_policy, model)
    sgd_optimizer.add_observer(train_loss_tracker)
    sgd_optimizer.add_observer(valid_loss_tracker)
    import time
    t = time.time()
    sgd_optimizer.optimize()
    from quagga.cuda import cudart
    cudart.cuda_device_synchronize()
    print time.time() - t