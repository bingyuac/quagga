import glob
import logging
import cPickle
import numpy as np
from quagga import Model
from itertools import chain
from itertools import islice
from quagga.utils import List
from quagga.cuda import cudart
from collections import Counter
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import DotBlock
from collections import defaultdict
from quagga.blocks import LstmBlock
from numpy.random import RandomState
from quagga.blocks import RepeatBlock
from quagga.connector import Connector
from quagga.optimizers import Optimizer
from quagga.blocks import SequencerBlock
from quagga.blocks import SoftmaxCeBlock
from quagga.blocks import RowSlicingBlock
from quagga.blocks import LastSelectorBlock
from quagga.blocks import ParameterContainer
from quagga.blocks import L2RegularizationBlock
from quagga.optimizers.observers import Validator
from quagga.optimizers.observers import Hdf5Saver
from quagga.utils.initializers import H5pyInitializer
from quagga.optimizers.policies import FixedValuePolicy
from quagga.optimizers.observers import TrainLossTracker
from quagga.optimizers.observers import ValidLossTracker
from quagga.optimizers.steps import NagStep, SparseSgdStep
from quagga.optimizers.stopping_criteria import MaxIterCriterion


def get_logger(file_name):
    logger = logging.getLogger('train_logger')
    handler = logging.FileHandler(file_name, mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%d-%m-%Y %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_dataset(max_sent_len=42):
    train_data = []
    valid_data = []

    rnd = RandomState(42)
    # for file_path in islice(sorted(glob.glob('data/valid/*'), key=lambda k: rnd.rand()), 0, 40):
    for file_path in islice(sorted(glob.glob('data/valid/*'), key=lambda k: rnd.rand()), 0, 1):
        print file_path
        with open(file_path) as f:
            for line in f:
                line = line.decode('utf-8').split()
                if len(line) < max_sent_len:
                    valid_data.append(line)

    print '===train===='
    # for file_path in islice(sorted(glob.glob('data/train/*'), key=lambda k: rnd.rand()), 0, 40):
    for file_path in islice(sorted(glob.glob('data/train/*'), key=lambda k: rnd.rand()), 0, 1):
        with open(file_path) as f:
            print file_path
            for line in f:
                line = line.decode('utf-8').split()
                if len(line) < max_sent_len:
                    train_data.append(line)

    # vocab = Counter()
    # for line in chain(train_data, valid_data):
    #     vocab.update(line)
    # print vocab.most_common(n=int(len(vocab) * 1.0))[-10:]
    # print vocab.most_common(n=int(len(vocab) * 0.085))[-10:]
    # vocab = set([e[0] for e in vocab.most_common(n=int(len(vocab) * 0.085))])
    # vocab.update(['<UNK>'])
    # word_to_idx = {}
    # idx_to_word = []
    # for i, word in enumerate(vocab):
    #     word_to_idx[word] = i
    #     idx_to_word.append(word)
    # word_to_idx['<<S>>'] = i + 1
    # idx_to_word.append('<<S>>')
    # unk_idx = word_to_idx['<UNK>']

    with open('vocab.pckl') as f:
        vocab = cPickle.load(f)
    word_to_idx = vocab['word_to_idx']
    idx_to_word = vocab['idx_to_word']
    unk_idx = word_to_idx['<UNK>']

    for i in xrange(len(train_data)):
        train_data[i] = [word_to_idx.get(e, unk_idx) for e in train_data[i]]
    for i in xrange(len(valid_data)):
        valid_data[i] = [word_to_idx.get(e, unk_idx) for e in valid_data[i]]

    return train_data, valid_data, word_to_idx, idx_to_word


class HomogeneousDataIterator(object):
    def __init__(self, data, batch_size, randomize=False, infinite=False):
        self.batch_size = batch_size
        self.data = defaultdict(list)
        for line in data:
            self.data[len(line)].append(line)
        if randomize:
            self.rng = np.random.RandomState(7823)
        self.infinite = infinite

    def __iter__(self):
        while True:
            for batch in self.__iter():
                yield batch
            if not self.infinite:
                break
            print 'epoch'

    def __iter(self):
        randomize = hasattr(self, 'rng')
        if randomize:
            for data in self.data.itervalues():
                self.rng.shuffle(data)
        progress = defaultdict(int)
        available_lengths = self.data.keys()
        if randomize:
            get_sent_len = lambda: self.rng.choice(available_lengths)
        else:
            get_sent_len = lambda: available_lengths[0]
        batch = []
        b_size = self.batch_size
        k = get_sent_len()
        while available_lengths:
            batch.extend(self.data[k][progress[k]:progress[k] + b_size])
            progress[k] += b_size
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                b_size = self.batch_size
                k = get_sent_len()
            else:
                b_size = self.batch_size - len(batch)
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
        if batch:
            yield batch


class DataBlock(object):
    def __init__(self, train_data, valid_data, batch_size, word_dropout_prob, device_id):
        self.train_data = HomogeneousDataIterator(train_data, batch_size, randomize=True, infinite=True)
        self.valid_data = HomogeneousDataIterator(valid_data, batch_size)
        self.train_data_iterator = iter(self.train_data)
        self.valid_data_iterator = iter(self.valid_data)
        self.word_keep_prob = 1.0 - word_dropout_prob
        self.rnd = RandomState(47571)
        self.unk_idx = word_to_idx['<UNK>']

        self.context = Context(device_id)
        c = Counter([len(line) for line in chain(train_data, valid_data)])
        print c.most_common()
        max_len = max([len(line) for line in chain(train_data, valid_data)])

        self.enc_x = Connector(Matrix.empty(batch_size, max_len, 'int', device_id))
        self.enc_lengths = Matrix.empty(self.enc_x.nrows, 1, 'int', device_id)
        self._enc_mask = Matrix.empty(self.enc_x.nrows, self.enc_x.ncols, 'float', device_id)
        self.enc_mask = List([Connector(self._enc_mask[:, i]) for i in xrange(max_len)], self.enc_x.ncols)

        self.dec_x = Connector(Matrix.empty(batch_size, max_len + 1, 'int', device_id))
        self._dec_y = Matrix.empty(batch_size, max_len + 1, 'int', device_id)
        self.dec_y = List([Connector(self._dec_y[:, i]) for i in xrange(max_len + 1)], self._dec_y.ncols)
        self.dec_lengths = Matrix.empty(self.dec_x.nrows, 1, 'int', device_id)
        self._dec_mask = Matrix.empty(self.dec_x.nrows, self.dec_x.ncols, 'float', device_id)
        self.dec_mask = List([Connector(self._dec_mask[:, i]) for i in xrange(max_len + 1)], self.dec_x.ncols)

        self.blocking_contexts = None
        self.training_mode = True

    def set_training_mode(self):
        self.training_mode = True

    def set_testing_mode(self):
        self.training_mode = False

    def fprop(self):
        if self.training_mode:
            data = next(self.train_data_iterator)
        else:
            try:
                data = next(self.valid_data_iterator)
            except StopIteration as e:
                self.valid_data_iterator = iter(self.valid_data)
                raise e
        lengths_npa = np.array([[len(e)] for e in data], np.int32, order='F')
        max_len = int(np.max(lengths_npa))

        self.enc_lengths.assign_npa(self.context, lengths_npa)
        self._enc_mask.mask_column_numbers_row_wise(self.context, self.enc_lengths)
        for e in self.enc_mask:
            e.last_modification_context = self.context

        lengths_npa += 1
        self.dec_lengths.assign_npa(self.context, lengths_npa)
        self._dec_mask.mask_column_numbers_row_wise(self.context, self.dec_lengths)
        for e in self.dec_mask:
            e.last_modification_context = self.context

        enc_x_npa = np.zeros((len(data), max_len), np.int32, 'F')
        dec_x_npa = np.zeros((len(data), max_len + 1), np.int32, 'F')
        dec_y_npa = np.zeros((len(data), max_len + 1), np.int32, 'F')
        for k, e in enumerate(data):
            enc_x_npa[k, :len(e)] = e
            if self.training_mode:
                new_e = [_ if self.rnd.rand() < self.word_keep_prob else self.unk_idx for _ in e]
            else:
                new_e = e
            dec_x_npa[k, :len(e) + 1] = [word_to_idx['<<S>>']] + new_e
            dec_y_npa[k, :len(e) + 1] = e + [word_to_idx['<<S>>']]
        self.enc_x.assign_npa(self.context, enc_x_npa)
        self.dec_x.assign_npa(self.context, dec_x_npa)
        self._dec_y.assign_npa(self.context, dec_y_npa)
        for e in self.dec_y:
            e.last_modification_context = self.context

        self.enc_mask.fprop()
        self.dec_mask.fprop()
        self.enc_x.fprop()
        self.dec_x.fprop()
        self.dec_y.fprop()


if __name__ == '__main__':
    train_data, valid_data, word_to_idx, idx_to_word = load_dataset()
    print 'word_to_idx', len(word_to_idx)
    print 'train_data', len(train_data)
    print 'valid_data', len(valid_data)
    with open('vocab.pckl', 'w') as f:
        cPickle.dump({'word_to_idx': word_to_idx,
                      'idx_to_word': idx_to_word}, f)
    model_file_name = 'drop_auto.hdf5'
    p = ParameterContainer(embd_W={'init': H5pyInitializer(model_file_name, 'embd_W'),
                                   'device_id': 0},
                           enc_lstm_c0={'init': H5pyInitializer(model_file_name, 'enc_lstm_c0'),
                                        'device_id': 0},
                           enc_lstm_h0={'init': H5pyInitializer(model_file_name, 'enc_lstm_h0'),
                                        'device_id': 0},
                           enc_lstm_W={'init': H5pyInitializer(model_file_name, 'enc_lstm_W'),
                                       'device_id': 0},
                           enc_lstm_R={'init': H5pyInitializer(model_file_name, 'enc_lstm_R'),
                                       'device_id': 0},
                           dec_lstm_c0={'init': H5pyInitializer(model_file_name, 'dec_lstm_c0'),
                                        'device_id': 0},
                           dec_lstm_W={'init': H5pyInitializer(model_file_name, 'dec_lstm_W'),
                                       'device_id': 0},
                           dec_lstm_R={'init': H5pyInitializer(model_file_name, 'dec_lstm_R'),
                                       'device_id': 0},
                           sce_dot_block_W={'init': H5pyInitializer(model_file_name, 'sce_dot_block_W'),
                                            'device_id': 0},
                           sce_dot_block_b={'init': H5pyInitializer(model_file_name, 'sce_dot_block_b'),
                                            'device_id': 0})
    data_block = DataBlock(train_data, valid_data, 64, word_dropout_prob=0.99, device_id=0)
    enc_embd_block = RowSlicingBlock(p['embd_W'], data_block.enc_x)
    enc_c_repeat_block = RepeatBlock(p['enc_lstm_c0'], data_block.enc_x.nrows, axis=0, device_id=0)
    enc_h_repeat_block = RepeatBlock(p['enc_lstm_h0'], data_block.enc_x.nrows, axis=0, device_id=0)
    enc_lstm_block = SequencerBlock(block_class=LstmBlock,
                                    params=[p['enc_lstm_W'], p['enc_lstm_R'], 0.25],
                                    sequences=[enc_embd_block.output, data_block.enc_mask],
                                    output_names=['h'],
                                    prev_names=['c', 'h'],
                                    paddings=[enc_c_repeat_block.output, enc_h_repeat_block.output],
                                    reverse=False,
                                    device_id=0)
    dec_embd_block = RowSlicingBlock(p['embd_W'], data_block.dec_x)
    dec_c_repeat_block = RepeatBlock(p['dec_lstm_c0'], data_block.enc_x.nrows, axis=0, device_id=0)
    last_selector_block = LastSelectorBlock(enc_lstm_block.h)
    l2_reg_block = L2RegularizationBlock(last_selector_block.output, 0.001)
    dec_lstm_block = SequencerBlock(block_class=LstmBlock,
                                    params=[p['dec_lstm_W'], p['dec_lstm_R'], 0.25],
                                    sequences=[dec_embd_block.output, data_block.dec_mask],
                                    output_names=['h'],
                                    prev_names=['c', 'h'],
                                    paddings=[dec_c_repeat_block.output, last_selector_block.output],
                                    reverse=False,
                                    device_id=0)
    seq_dot_block = SequencerBlock(block_class=DotBlock,
                                   params=[p['sce_dot_block_W'], p['sce_dot_block_b']],
                                   sequences=[dec_lstm_block.h],
                                   output_names=['output'],
                                   device_id=0)
    seq_sce_block = SequencerBlock(block_class=SoftmaxCeBlock,
                                   params=[],
                                   sequences=[seq_dot_block.output, data_block.dec_y, data_block.dec_mask],
                                   output_names=[],
                                   device_id=0)
    model = Model([p, data_block,
                   enc_embd_block,
                   enc_c_repeat_block, enc_h_repeat_block, enc_lstm_block,
                   dec_embd_block,
                   dec_c_repeat_block, last_selector_block, dec_lstm_block,
                   l2_reg_block,
                   seq_dot_block, seq_sce_block])
    print 'go'

    logger = get_logger('train.log')
    train_loss_tracker = TrainLossTracker(model, 200, logger)
    valid_loss_tracker = ValidLossTracker(logger)
    validator = Validator(model, 16000)
    validator.add_observer(valid_loss_tracker)
    saver = Hdf5Saver(p.trainable_parameters, 2000, 'drop_auto.hdf5', logger)

    trainable_parameters = dict(p.trainable_parameters)
    sparse_sgd_step = SparseSgdStep([trainable_parameters['embd_W']], FixedValuePolicy(0.01))
    del trainable_parameters['embd_W']
    nag_step = NagStep(trainable_parameters.values(), FixedValuePolicy(0.01), FixedValuePolicy(0.9))
    data_block.blocking_contexts = nag_step.blocking_contexts + sparse_sgd_step.blocking_contexts


    class DependencySetter(object):
        def notify(self):
            data_block.blocking_contexts = nag_step.blocking_contexts + sparse_sgd_step.blocking_contexts


    criterion = MaxIterCriterion(2000000)
    optimizer = Optimizer(criterion, model)
    optimizer.add_observer(sparse_sgd_step)
    optimizer.add_observer(nag_step)
    optimizer.add_observer(DependencySetter())
    optimizer.add_observer(train_loss_tracker)
    optimizer.add_observer(validator)
    optimizer.add_observer(saver)
    optimizer.add_observer(criterion)
    optimizer.optimize()

    for device_id in xrange(cudart.cuda_get_device_count()):
        cudart.cuda_set_device(device_id)
        cudart.cuda_device_synchronize()