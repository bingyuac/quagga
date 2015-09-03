import os
import gzip
import json
import cPickle
import logging
import numpy as np
from quagga import Model
from itertools import izip
from urllib import urlretrieve
from quagga.matrix import Matrix
from quagga.context import Context
from collections import defaultdict
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


class HomogeneousDataGenerator(object):
    def __init__(self, sentences, batch_size, sentence_max_len, randomize=False, infinite=False):
        sentences = [s for s in sentences if len(s) <= sentence_max_len]
        self.b_size = batch_size
        self.flatten_sentences = [w for s in sentences for w in s]
        self.offsets = defaultdict(list)
        for s_offsets in self.__get_sentence_offsets(sentences):
            self.offsets[s_offsets[1] - s_offsets[0]].append(s_offsets)
        if randomize:
            self.rng = np.random.RandomState(42)
        self.infinite = infinite

    def __iter__(self):
        while True:
            for batch_offsets in self.__iter():
                yield batch_offsets
            if not self.infinite:
                break

    def __iter(self):
        randomize = hasattr(self, 'rng')
        if randomize:
            for s_offsets in self.offsets.itervalues():
                self.rng.shuffle(s_offsets)
        batch_idx = defaultdict(int)
        available_lengths = range(len(self.offsets))
        batch_offsets = []
        b_size = self.b_size
        while available_lengths:
            k = self.rng.choice(available_lengths) if randomize else available_lengths[0]
            i = batch_idx[k]
            batch_offsets.append(self.offsets[k][b_size * i:b_size * (i + 1)])
            if len(batch_offsets) == self.b_size:
                yield batch_offsets
                batch_offsets = []
                b_size = self.b_size
                batch_idx[k] += 1
            else:
                available_lengths.remove(k)
                b_size = self.b_size - len(batch_offsets)
        if batch_offsets:
            yield batch_offsets

    @staticmethod
    def __get_sentence_offsets(sentences):
        sentence_offsets = []
        offset = 0
        for s in sentences:
            sentence_offsets.append((offset, offset + len(s)))
            offset += len(s)
        return sentence_offsets


class SentencesMiniBatchesGenerator(object):
    def __init__(self, ptb_train, ptb_valid, batch_size, sentence_max_len, device_id):
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.train_offsets = HomogeneousDataGenerator(ptb_train, batch_size, sentence_max_len, randomize=True, infinite=True)
        self.valid_offsets = HomogeneousDataGenerator(ptb_valid, batch_size, sentence_max_len)

        train_sentences = np.array([self.train_offsets.flatten_sentences])
        valid_sentences = np.array([self.valid_offsets.flatten_sentences])
        self.train_sents = Matrix.from_npa(train_sentences, 'int', device_id)
        self.valid_sents = Matrix.from_npa(valid_sentences, 'int', device_id)

        self.sentence_batch = Matrix.empty(batch_size, sentence_max_len, 'int', device_id)
        self.sentence_batch.sync_fill(0)
        self.mask = Matrix.empty(batch_size, sentence_max_len, 'float', device_id)

        self.train_offsets_iterator = iter(self.train_offsets)
        self.valid_offsets_iterator = iter(self.valid_offsets)
        self.training_mode = True

    def fprop(self):
        if self.training_mode:
            offsets = next(self.train_offsets_iterator)
            sents = self.train_sents
        else:
            try:
                offsets = next(self.valid_offsets_iterator)
                sents = self.valid_sents
            except StopIteration as e:
                self.valid_offsets_iterator = iter(self.valid_offsets)
                raise e

        self.mask.fill(self.context, 1.0)
        self.context.wait(self.blocking_context)
        for k, offset in enumerate(offsets):
            sent = sents[:, offset[0]:offset[1]]
            sent_len = offset[1] - offset[0]
            batch_chunk = self.sentence_batch[k, :sent_len]
            batch_chunk.copy(self.context, sent)
            if sent_len != self.sentence_batch.nrows:
                sub_mask = self.mask[sent_len:, k]
                sub_mask.fill(self.context, 0.0)

    def set_training_mode(self):
        self.training_mode = True

    def set_testing_mode(self):
        self.training_mode = False


if __name__ == '__main__':
    ptb_train, ptb_valid, ptb_test, vocab = load_ptb_dataset()
    SentencesMiniBatchesGenerator
