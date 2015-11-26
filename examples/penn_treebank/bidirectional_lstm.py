import os
import logging
import numpy as np
from quagga import Model
from quagga.utils import List
from quagga.cuda import cudart
from urllib import urlretrieve
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import DotBlock
from collections import defaultdict
from quagga.blocks import LstmBlock
from quagga.blocks import RepeatBlock
from quagga.connector import Connector
from quagga.optimizers import Optimizer
from quagga.blocks import SequencerBlock
from quagga.blocks import SoftmaxCeBlock
from quagga.blocks import RowSlicingBlock
from quagga.blocks import ParameterContainer
from quagga.utils.initializers import Constant
from quagga.blocks import HorizontalStackBlock
from quagga.utils.initializers import Orthogonal
from quagga.optimizers.observers import Hdf5Saver
from quagga.optimizers.observers import ValidLossTracker
from quagga.optimizers.observers import TrainLossTracker
from quagga.optimizers.policies import FixedValuePolicy
from quagga.optimizers.policies import FixedValuePolicy
from quagga.optimizers.stopping_criteria import MaxIterCriterion
from quagga.optimizers.steps import NagStep, SparseSgdStep, SgdStep


def get_logger(file_name):
    logger = logging.getLogger('train_logger')
    handler = logging.FileHandler(file_name, mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%d-%m-%Y %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_ptb_dataset():
    train_file_path = 'ptb_train.txt'
    valid_file_path = 'ptb_valid.txt'
    test_file_path = 'ptb_test.txt'

    if not os.path.exists(train_file_path):
        urlretrieve('https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt', train_file_path)
    if not os.path.exists(valid_file_path):
        urlretrieve('https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt', valid_file_path)
    if not os.path.exists(test_file_path):
        urlretrieve('https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt', test_file_path)

    vocab = {}
    idx_to_word = []
    ptb_train = []
    with open(train_file_path) as f:
        for line in f:
            sentence = ['<S>'] + line.strip().split() + ['</S>']
            for word in sentence:
                if word not in vocab:
                    vocab[word] = len(idx_to_word)
                    idx_to_word.append(word)
            ptb_train.append([vocab[word] for word in sentence])

    ptb_valid = []
    with open(valid_file_path) as f:
        for line in f:
            sentence = ['<S>'] + line.strip().split() + ['</S>']
            ptb_valid.append([vocab[word] for word in sentence])

    ptb_test = []
    with open(test_file_path) as f:
        for line in f:
            sentence = ['<S>'] + line.strip().split() + ['</S>']
            ptb_test.append([vocab[word] for word in sentence])

    return ptb_train, ptb_valid, ptb_test, vocab, idx_to_word


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
        progress = defaultdict(int)
        available_lengths = self.offsets.keys()
        if randomize:
            get_sent_len = lambda: self.rng.choice(available_lengths)
        else:
            get_sent_len = lambda: available_lengths[0]
        batch_offsets = []
        b_size = self.b_size
        k = get_sent_len()
        while available_lengths:
            batch_offsets.extend(self.offsets[k][progress[k]:progress[k]+b_size])
            progress[k] += b_size
            if len(batch_offsets) == self.b_size:
                yield batch_offsets
                batch_offsets = []
                b_size = self.b_size
                k = get_sent_len()
            else:
                b_size = self.b_size - len(batch_offsets)
                i = available_lengths.index(k)
                del available_lengths[i]
                if not available_lengths:
                    break
                if i == 0:
                    k = available_lengths[0]
                elif i >= len(available_lengths) - 1:
                    k = available_lengths[-1]
                else:
                    k = available_lengths[i + self.rng.choice([-1, 1])]
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


class PtbMiniBatchesGenerator(object):
    def __init__(self, ptb_train, ptb_valid, batch_size, sentence_max_len, device_id):
        self.blocking_contexts = None
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.train_offsets = HomogeneousDataGenerator(ptb_train, batch_size, sentence_max_len, randomize=True, infinite=True)
        self.valid_offsets = HomogeneousDataGenerator(ptb_valid, batch_size, sentence_max_len)

        train_sentences = np.array([self.train_offsets.flatten_sentences])
        valid_sentences = np.array([self.valid_offsets.flatten_sentences])
        self.train_sents = Matrix.from_npa(train_sentences, 'int', device_id)
        self.valid_sents = Matrix.from_npa(valid_sentences, 'int', device_id)
        self._sent_lengths = np.empty((batch_size, 1), dtype=np.int32, order='F')[...]
        self.sent_lengths = Matrix.from_npa(self._sent_lengths, device_id=device_id)

        sentence_batch = Matrix.empty(batch_size, sentence_max_len, 'int', device_id)
        self.sentence_batch = Connector(sentence_batch, self.context)
        self.sentence_batch.sync_fill(0)

        self._mask = Matrix.empty(sentence_batch.nrows, self.sentence_batch.ncols, 'float', device_id)
        self.mask = List([Connector(self._mask[:, i]) for i in xrange(sentence_max_len)], self.sentence_batch.ncols)
        self.train_offsets_iterator = iter(self.train_offsets)
        self.valid_offsets_iterator = iter(self.valid_offsets)
        self.training_mode = True

    def set_training_mode(self):
        self.training_mode = True

    def set_testing_mode(self):
        self.training_mode = False

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
        self.context.wait(*self.blocking_contexts)
        self._sent_lengths = self._sent_lengths.base[:len(offsets)]
        self.sentence_batch.nrows = len(offsets)
        for k, offset in enumerate(offsets):
            self.sentence_batch[k].assign(self.context, sents[:, offset[0]:offset[1]])
            self._sent_lengths[k] = offset[1] - offset[0]
        max_sent_len = int(np.max(self._sent_lengths))
        self.sentence_batch.last_modification_context = self.context
        self.sentence_batch.ncols = max_sent_len
        self.sent_lengths.assign_npa(self.context, self._sent_lengths)
        self._mask.mask_column_numbers_row_wise(self.context, self.sent_lengths)
        for e in self.mask:
            e.last_modification_context = self.context
        self.sentence_batch.fprop()
        self.mask.fprop()


if __name__ == '__main__':
    ptb_train, ptb_valid, ptb_test, vocab, idx_to_word = load_ptb_dataset()
    print len(ptb_train), len(ptb_valid), len(ptb_test), len(idx_to_word)
    get_orth_W = Orthogonal(256, 512)
    get_orth_R = Orthogonal(512, 512)
    p = ParameterContainer(embd_W={'init': Orthogonal(len(vocab), 256),
                                   'device_id': 0},
                           lstm_fwd_c0={'init': Constant(1, 512),
                                        'device_id': 0,
                                        'trainable': False},
                           lstm_fwd_h0={'init': Constant(1, 512),
                                        'device_id': 0,
                                        'trainable': False},
                           lstm_fwd_W={'init': lambda: np.hstack((get_orth_W(), get_orth_W(), get_orth_W(), get_orth_W())),
                                       'device_id': 0},
                           lstm_fwd_R={'init': lambda: np.hstack((get_orth_R(), get_orth_R(), get_orth_R(), get_orth_R())),
                                       'device_id': 0},
                           lstm_bwd_c0={'init': Constant(1, 512),
                                        'device_id': 0,
                                        'trainable': False},
                           lstm_bwd_h0={'init': Constant(1, 512),
                                        'device_id': 0,
                                        'trainable': False},
                           lstm_bwd_W={'init': lambda: np.hstack((get_orth_W(), get_orth_W(), get_orth_W(), get_orth_W())),
                                       'device_id': 0},
                           lstm_bwd_R={'init': lambda: np.hstack((get_orth_R(), get_orth_R(), get_orth_R(), get_orth_R())),
                                       'device_id': 0},
                           sce_dot_block_W={'init': Orthogonal(1024, len(vocab)),
                                            'device_id': 0},
                           sce_dot_block_b={'init': Constant(1, len(vocab)),
                                            'device_id': 0})
    data_block = PtbMiniBatchesGenerator(ptb_train, ptb_valid, batch_size=64, sentence_max_len=100, device_id=0)
    seq_embd_block = RowSlicingBlock(p['embd_W'], data_block.sentence_batch)
    # remove last in the list
    output = List(seq_embd_block.output[:-1], seq_embd_block.output.length - 1)
    c_fwd_repeat_block = RepeatBlock(p['lstm_fwd_c0'], data_block.sentence_batch.nrows, axis=0, device_id=0)
    h_fwd_repeat_block = RepeatBlock(p['lstm_fwd_h0'], data_block.sentence_batch.nrows, axis=0, device_id=0)
    fwd_lstm_block = SequencerBlock(block_class=LstmBlock,
                                    params=[p['lstm_fwd_W'], p['lstm_fwd_R'], 0.5],
                                    sequences=[output, data_block.mask],
                                    output_names=['h'],
                                    prev_names=['c', 'h'],
                                    paddings=[c_fwd_repeat_block.output, h_fwd_repeat_block.output],
                                    reverse=False,
                                    device_id=0)
    # remove first in the list
    output = List(seq_embd_block.output[1:], seq_embd_block.output.length - 1)
    c_bwd_repeat_block = RepeatBlock(p['lstm_bwd_c0'], data_block.sentence_batch.nrows, axis=0, device_id=0)
    h_bwd_repeat_block = RepeatBlock(p['lstm_bwd_h0'], data_block.sentence_batch.nrows, axis=0, device_id=0)
    bwd_lstm_block = SequencerBlock(block_class=LstmBlock,
                                    params=[p['lstm_bwd_W'], p['lstm_bwd_R'], 0.5],
                                    sequences=[output, data_block.mask],
                                    output_names=['h'],
                                    prev_names=['c', 'h'],
                                    paddings=[c_bwd_repeat_block.output, h_bwd_repeat_block.output],
                                    reverse=True,
                                    device_id=0)
    seq_hstack = SequencerBlock(block_class=HorizontalStackBlock,
                                params=[],
                                sequences=[List([h_fwd_repeat_block.output] + fwd_lstm_block.h[:], fwd_lstm_block.h.length + 1),
                                           List(bwd_lstm_block.h[:] + [h_bwd_repeat_block.output], bwd_lstm_block.h.length + 1)],
                                output_names=['output'],
                                device_id=0)
    seq_dot_block = SequencerBlock(block_class=DotBlock,
                                   params=[p['sce_dot_block_W'], p['sce_dot_block_b']],
                                   sequences=[seq_hstack.output],
                                   output_names=['output'],
                                   device_id=0)
    sentence_batch = List([Connector(data_block.sentence_batch[:, i]) for i in xrange(data_block.sentence_batch.ncols)], data_block.sentence_batch.ncols)
    seq_sce_block = SequencerBlock(block_class=SoftmaxCeBlock,
                                   params=[],
                                   sequences=[seq_dot_block.output, sentence_batch, data_block.mask],
                                   device_id=0)
    model = Model([p, data_block, seq_embd_block,
                   c_fwd_repeat_block, h_fwd_repeat_block, fwd_lstm_block,
                   c_bwd_repeat_block, h_bwd_repeat_block, bwd_lstm_block,
                   seq_hstack, seq_dot_block, seq_sce_block])

    logger = get_logger('train.log')
    momentum_policy = FixedValuePolicy(0.95)
    train_loss_tracker = TrainLossTracker(model, 100, logger)
    valid_loss_tracker = ValidLossTracker(model, 500, logger)
    saver = Hdf5Saver(p.trainable_parameters, 5000, 'ptb_parameters.hdf5', logger)
    trainable_parameters = dict(p.trainable_parameters)
    sparse_sgd_step = SparseSgdStep([trainable_parameters['embd_W']], FixedValuePolicy(0.01))
    del trainable_parameters['embd_W']
    nag_step = NagStep(trainable_parameters.values(), FixedValuePolicy(0.01), momentum_policy)
    # nag_step = SgdStep(trainable_parameters.values(), learning_rate_policy)
    data_block.blocking_contexts = nag_step.blocking_contexts + sparse_sgd_step.blocking_contexts
    criterion = MaxIterCriterion(20000)

    optimizer = Optimizer(criterion, model)
    optimizer.add_observer(sparse_sgd_step)
    optimizer.add_observer(nag_step)
    optimizer.add_observer(train_loss_tracker)
    optimizer.add_observer(valid_loss_tracker)
    optimizer.add_observer(saver)
    optimizer.add_observer(criterion)
    optimizer.optimize()

    for device_id in xrange(cudart.cuda_get_device_count()):
        cudart.cuda_set_device(device_id)
        cudart.cuda_device_synchronize()