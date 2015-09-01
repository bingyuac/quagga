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
from quagga.optimizers.observers import Saver
from sklearn.preprocessing import OneHotEncoder
from quagga.optimizers.observers import ValidLossTracker
from quagga.optimizers.observers import TrainLossTracker
from quagga.optimizers.policies import FixedLearningRatePolicy
from quagga.optimizers.stopping_criteria import MaxIterCriterion


def load_ptb_dataset():
    train_file_path = 'ptb_train.txt'
    valid_file_path = 'ptb_valid.txt'
    test_file_path = 'ptb_test.txt'

    if not os.path.exists(train_file_path):
        urlretrieve('https://github.com/wojzaremba/lstm/raw/master/data/ptb.train.txt', train_file_path)
    if not os.path.exists(valid_file_path):
        urlretrieve('https://github.com/wojzaremba/lstm/raw/master/data/ptb.valid.txt', valid_file_path)
    if not os.path.exists(test_file_path):
        urlretrieve('https://github.com/wojzaremba/lstm/raw/master/data/ptb.test.txt', test_file_path)

    max_word_idx = 0
    vocab = {}
    ptb_train = []
    with open(train_file_path) as f:
        for line in f:
            sentence = line.strip().split()
            for word in sentence:
                if word not in vocab:
                    vocab[word] = max_word_idx
                    max_word_idx += 1
            ptb_train.append([vocab[word] for word in sentence])

    with open(valid_file_path) as f:
        ptb_valid = [[vocab[word] for word in line.strip().split()] for line in f]

    with open(test_file_path) as f:
        ptb_test = [[vocab[word] for word in line.strip().split()] for line in f]

    return ptb_train, ptb_valid, ptb_test, vocab


class SentencesMiniBatchesGenerator(object):
    def __init__(self, ptb_train, ptb_valid, batch_size, sentence_max_length, device_id):
        self.ptb_train = ptb_train
        self.ptb_valid = ptb_valid
        self.batch_size = batch_size
        self.sentence_max_length = sentence_max_length
        self.device_id = device_id

        self.train_sentences = Matrix.empty(1, len([w for s in ptb_train for w in s]), 'int', device_id)
        self.valid_sentences = Matrix.empty(1, len([w for s in ptb_valid for w in s]), 'int', device_id)

        self.sentence_batch = Matrix.empty(batch_size, sentence_max_length, 'int', device_id)


        # self.h


        self.training_mode = True

    def set_training_mode(self):
        self.training_mode = True

    def set_testing_mode(self):
        self.training_mode = False

    def fprop(self):
        pass


if __name__ == '__main__':
    ptb_train, ptb_valid, ptb_test, vocab = load_ptb_dataset()