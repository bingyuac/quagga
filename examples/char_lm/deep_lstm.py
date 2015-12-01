import glob
import cPickle
import logging
import numpy as np
from quagga import Model
from quagga.utils import List
from quagga.cuda import cudart
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import DotBlock
from quagga.blocks import LstmBlock
from collections import defaultdict
from quagga.blocks import RepeatBlock
from quagga.connector import Connector
from quagga.optimizers import Optimizer
from quagga.blocks import SequencerBlock
from quagga.blocks import SoftmaxCeBlock
from quagga.blocks import RowSlicingBlock
from quagga.optimizers.steps import NagStep
from quagga.blocks import ParameterContainer
from quagga.utils.initializers import Constant
from quagga.utils.initializers import Orthogonal
from quagga.optimizers.steps import SparseSgdStep
from quagga.optimizers.observers import Hdf5Saver
from quagga.utils.initializers import H5pyInitializer
from quagga.optimizers.policies import FixedValuePolicy
from quagga.optimizers.observers import TrainLossTracker
from quagga.optimizers.policies import ScheduledValuePolicy
from quagga.optimizers.stopping_criteria import MaxIterCriterion


def get_logger(file_name):
    logger = logging.getLogger('train_logger')
    # handler = logging.FileHandler(file_name, mode='w')
    handler = logging.FileHandler(file_name)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%d-%m-%Y %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def fancy_chunker(line, size, min_size):
    line = line.split()
    n = 0
    chunk = []
    k = len(line)
    i = 0
    while i < k:
        e = line[i]
        if len(e) <= size - n - (1 if n else 0):
            chunk.append(e)
            n += len(e) + (1 if n else 0)
            i += 1
        else:
            if chunk:
                chunk = u' '.join(chunk)
                if len(chunk) >= min_size:
                    yield chunk
                else:
                    pass
                    # print 'I skipped this because it is too short', chunk
                chunk = []
                n = 0
            else:
                i += 1
                # print 'I skipped this because it is too long', e
    if chunk:
        chunk = u' '.join(chunk)
        if len(chunk) >= min_size:
            yield chunk
        else:
            pass
            # print 'I skipped this because it is too short', chunk


def load_dataset():
    n = 420
    data = []
    char_to_idx = {}
    idx_to_char = []

    for file_path in glob.glob('data/clean/*.txt'):
        with open(file_path) as f:
            for line in f:
                line = line.decode('utf-8')
                for sub_line in fancy_chunker(line, n, 2):
                    data.append(sub_line)
    vocab = set()
    for sub_line in data:
        vocab.update(sub_line)
    for i, char in enumerate(vocab):
        char_to_idx[char] = i
        idx_to_char.append(char)
    char_to_idx['<unk>'] = i + 1
    idx_to_char.append('<unk>')

    char_data = []
    for sub_line in data:
        char_data.append([char_to_idx[char] for char in sub_line])
    return char_data, char_to_idx, idx_to_char


class HomogeneousDataIterator(object):
    def __init__(self, data, char_to_idx, batch_size, randomize=False, infinite=False):
        self.char_to_idx = char_to_idx
        self.batch_size = batch_size
        self.data = defaultdict(list)
        for sub_line in data:
            self.data[len(sub_line)].append(sub_line)
        if randomize:
            self.rng = np.random.RandomState(42)
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
    def __init__(self, data, char_to_idx, batch_size, x_device_id, y_device_id):
        self.data = HomogeneousDataIterator(data, char_to_idx, batch_size, True, True)
        self.data_iterator = iter(self.data)
        self.x_context = Context(x_device_id)
        self.y_context = Context(y_device_id)
        max_len = 0
        for sub_line in data:
            cur_len = len(sub_line)
            if cur_len > max_len:
                max_len = cur_len
        print max_len
        self.x = Connector(Matrix.empty(batch_size, max_len - 1, 'int', x_device_id))
        self._y = Matrix.empty(batch_size, max_len - 1, 'int', y_device_id)
        self.y = List([Connector(self._y[:, i]) for i in xrange(max_len - 1)], self.x.ncols)
        self.lengths = Matrix.empty(self.x.nrows, 1, 'int', x_device_id)
        self._mask = Matrix.empty(self.x.nrows, self.x.ncols, 'float', x_device_id)
        self.mask = List([Connector(self._mask[:, i]) for i in xrange(max_len)], self.x.ncols)
        self.blocking_contexts = None

    def fprop(self):
        self.x_context.wait(*self.blocking_contexts)
        self.y_context.wait(*self.blocking_contexts)
        data = next(self.data_iterator)
        lengths_npa = np.array([[len(e) - 1] for e in data], np.int32, order='F')
        x_npa = np.zeros((len(data), int(np.max(lengths_npa))), np.int32, 'F')
        for k, e in enumerate(data):
            x_npa[k, :len(e) - 1] = e[:-1]
        self.x.assign_npa(self.x_context, x_npa)
        y_npa = np.zeros((len(data), int(np.max(lengths_npa))), np.int32, 'F')
        for k, e in enumerate(data):
            y_npa[k, :len(e) - 1] = e[1:]
        self._y.assign_npa(self.y_context, y_npa)
        for e in self.y:
            e.last_modification_context = self.y_context
        self.lengths.assign_npa(self.x_context, lengths_npa)
        self._mask.mask_column_numbers_row_wise(self.x_context, self.lengths)
        for e in self.mask:
            e.last_modification_context = self.x_context
        self.x.fprop()
        self.y.fprop()
        self.mask.fprop()


if __name__ == '__main__':
    char_data, char_to_idx, idx_to_char = load_dataset()
    with open('vocab.pckl', 'w') as f:
        cPickle.dump({'char_to_idx': char_to_idx,
                      'idx_to_char': idx_to_char}, f)
    print len(char_data)
    print len(char_to_idx)

    model_file_name = 'ukr_char_lstm.hdf5'
    p = ParameterContainer(embd_W={'init': H5pyInitializer(model_file_name, 'embd_W'),
                                   'device_id': 1},
                           f_lstm_c0={'init': H5pyInitializer(model_file_name, 'f_lstm_c0'),
                                      'device_id': 1},
                           f_lstm_h0={'init': H5pyInitializer(model_file_name, 'f_lstm_h0'),
                                      'device_id': 1},
                           f_lstm_W={'init': H5pyInitializer(model_file_name, 'f_lstm_W'),
                                     'device_id': 1},
                           f_lstm_R={'init': H5pyInitializer(model_file_name, 'f_lstm_R'),
                                     'device_id': 1},
                           s_lstm_c0={'init': H5pyInitializer(model_file_name, 's_lstm_c0'),
                                      'device_id': 1},
                           s_lstm_h0={'init': H5pyInitializer(model_file_name, 's_lstm_h0'),
                                      'device_id': 1},
                           s_lstm_W={'init': H5pyInitializer(model_file_name, 's_lstm_W'),
                                     'device_id': 1},
                           s_lstm_R={'init': H5pyInitializer(model_file_name, 's_lstm_R'),
                                     'device_id': 1},
                           t_lstm_c0={'init': H5pyInitializer(model_file_name, 't_lstm_c0'),
                                      'device_id': 0},
                           t_lstm_h0={'init': H5pyInitializer(model_file_name, 't_lstm_h0'),
                                      'device_id': 0},
                           t_lstm_W={'init': H5pyInitializer(model_file_name, 't_lstm_W'),
                                     'device_id': 0},
                           t_lstm_R={'init': H5pyInitializer(model_file_name, 't_lstm_R'),
                                     'device_id': 0},
                           ft_lstm_c0={'init': H5pyInitializer(model_file_name, 'ft_lstm_c0'),
                                       'device_id': 0},
                           ft_lstm_h0={'init': H5pyInitializer(model_file_name, 'ft_lstm_h0'),
                                       'device_id': 0},
                           ft_lstm_W={'init': H5pyInitializer(model_file_name, 'ft_lstm_W'),
                                      'device_id': 0},
                           ft_lstm_R={'init': H5pyInitializer(model_file_name, 'ft_lstm_R'),
                                      'device_id': 0},
                           ff_lstm_c0={'init': H5pyInitializer(model_file_name, 'ff_lstm_c0'),
                                       'device_id': 0},
                           ff_lstm_h0={'init': H5pyInitializer(model_file_name, 'ff_lstm_h0'),
                                       'device_id': 0},
                           ff_lstm_W={'init': H5pyInitializer(model_file_name, 'ff_lstm_W'),
                                      'device_id': 0},
                           ff_lstm_R={'init': H5pyInitializer(model_file_name, 'ff_lstm_R'),
                                      'device_id': 0},
                           sce_dot_block_W={'init': H5pyInitializer(model_file_name, 'sce_dot_block_W'),
                                            'device_id': 0},
                           sce_dot_block_b={'init': H5pyInitializer(model_file_name, 'sce_dot_block_b'),
                                            'device_id': 0})
    # get_orth_W = Orthogonal(256, 1024)
    # get_stacked_orth_W = lambda: np.hstack((get_orth_W(), get_orth_W(), get_orth_W(), get_orth_W()))
    # get_orth_R = Orthogonal(1024, 1024)
    # get_stacked_orth_R = lambda: np.hstack((get_orth_R(), get_orth_R(), get_orth_R(), get_orth_R()))
    # p = ParameterContainer(embd_W={'init': Orthogonal(len(idx_to_char), 256),
    #                                'device_id': 1},
    #                        f_lstm_c0={'init': Constant(1, 1024),
    #                                   'device_id': 1},
    #                        f_lstm_h0={'init': Constant(1, 1024),
    #                                   'device_id': 1},
    #                        f_lstm_W={'init': get_stacked_orth_W,
    #                                  'device_id': 1},
    #                        f_lstm_R={'init': get_stacked_orth_R,
    #                                  'device_id': 1},
    #                        s_lstm_c0={'init': Constant(1, 1024),
    #                                   'device_id': 1},
    #                        s_lstm_h0={'init': Constant(1, 1024),
    #                                   'device_id': 1},
    #                        s_lstm_W={'init': get_stacked_orth_R,
    #                                  'device_id': 1},
    #                        s_lstm_R={'init': get_stacked_orth_R,
    #                                  'device_id': 1},
    #                        t_lstm_c0={'init': Constant(1, 1024),
    #                                   'device_id': 0},
    #                        t_lstm_h0={'init': Constant(1, 1024),
    #                                   'device_id': 0},
    #                        t_lstm_W={'init': get_stacked_orth_R,
    #                                  'device_id': 0},
    #                        t_lstm_R={'init': get_stacked_orth_R,
    #                                  'device_id': 0},
    #                        sce_dot_block_W={'init': Orthogonal(1024, len(idx_to_char)),
    #                                         'device_id': 0},
    #                        sce_dot_block_b={'init': Constant(1, len(idx_to_char)),
    #                                         'device_id': 0})
    data_block = DataBlock(char_data, char_to_idx, 50, x_device_id=1, y_device_id=0)
    embd_block = RowSlicingBlock(W=p['embd_W'], row_indexes=data_block.x)
    f_c_repeat_block = RepeatBlock(p['f_lstm_c0'], data_block.x.nrows, axis=0, device_id=1)
    f_h_repeat_block = RepeatBlock(p['f_lstm_h0'], data_block.x.nrows, axis=0, device_id=1)
    f_lstm_rnn_block = SequencerBlock(block_class=LstmBlock,
                                      params=[p['f_lstm_W'], p['f_lstm_R'], None],
                                      sequences=[embd_block.output, data_block.mask],
                                      output_names=['h'],
                                      prev_names=['c', 'h'],
                                      paddings=[f_c_repeat_block.output, f_h_repeat_block.output],
                                      reverse=False,
                                      device_id=1)
    s_c_repeat_block = RepeatBlock(p['s_lstm_c0'], data_block.x.nrows, axis=0, device_id=1)
    s_h_repeat_block = RepeatBlock(p['s_lstm_h0'], data_block.x.nrows, axis=0, device_id=1)
    s_lstm_rnn_block = SequencerBlock(block_class=LstmBlock,
                                      params=[p['s_lstm_W'], p['s_lstm_R'], None],
                                      sequences=[f_lstm_rnn_block.h, data_block.mask],
                                      output_names=['h'],
                                      prev_names=['c', 'h'],
                                      paddings=[s_c_repeat_block.output, s_h_repeat_block.output],
                                      reverse=False,
                                      device_id=1)
    t_c_repeat_block = RepeatBlock(p['t_lstm_c0'], data_block.x.nrows, axis=0, device_id=0)
    t_h_repeat_block = RepeatBlock(p['t_lstm_h0'], data_block.x.nrows, axis=0, device_id=0)
    t_lstm_rnn_block = SequencerBlock(block_class=LstmBlock,
                                      params=[p['t_lstm_W'], p['t_lstm_R'], None],
                                      sequences=[s_lstm_rnn_block.h, data_block.mask],
                                      output_names=['h'],
                                      prev_names=['c', 'h'],
                                      paddings=[t_c_repeat_block.output, t_h_repeat_block.output],
                                      reverse=False,
                                      device_id=0)
    ft_c_repeat_block = RepeatBlock(p['ft_lstm_c0'], data_block.x.nrows, axis=0, device_id=0)
    ft_h_repeat_block = RepeatBlock(p['ft_lstm_h0'], data_block.x.nrows, axis=0, device_id=0)
    ft_lstm_rnn_block = SequencerBlock(block_class=LstmBlock,
                                       params=[p['ft_lstm_W'], p['ft_lstm_R'], None],
                                       sequences=[t_lstm_rnn_block.h, data_block.mask],
                                       output_names=['h'],
                                       prev_names=['c', 'h'],
                                       paddings=[ft_c_repeat_block.output, ft_h_repeat_block.output],
                                       reverse=False,
                                       device_id=0)
    ff_c_repeat_block = RepeatBlock(p['ff_lstm_c0'], data_block.x.nrows, axis=0, device_id=0)
    ff_h_repeat_block = RepeatBlock(p['ff_lstm_h0'], data_block.x.nrows, axis=0, device_id=0)
    ff_lstm_rnn_block = SequencerBlock(block_class=LstmBlock,
                                       params=[p['ff_lstm_W'], p['ff_lstm_R'], None],
                                       sequences=[ft_lstm_rnn_block.h, data_block.mask],
                                       output_names=['h'],
                                       prev_names=['c', 'h'],
                                       paddings=[ff_c_repeat_block.output, ff_h_repeat_block.output],
                                       reverse=False,
                                       device_id=0)
    seq_dot_block = SequencerBlock(block_class=DotBlock,
                                   params=[p['sce_dot_block_W'], p['sce_dot_block_b']],
                                   sequences=[ff_lstm_rnn_block.h],
                                   output_names=['output'],
                                   device_id=0)
    seq_sce_block = SequencerBlock(block_class=SoftmaxCeBlock,
                                   params=[],
                                   sequences=[seq_dot_block.output, data_block.y, data_block.mask],
                                   output_names=[],
                                   device_id=0)
    model = Model([p, data_block, embd_block,
                   f_c_repeat_block, f_h_repeat_block, f_lstm_rnn_block,
                   s_c_repeat_block, s_h_repeat_block, s_lstm_rnn_block,
                   t_c_repeat_block, t_h_repeat_block, t_lstm_rnn_block,
                   ft_c_repeat_block, ft_h_repeat_block, ft_lstm_rnn_block,
                   ff_c_repeat_block, ff_h_repeat_block, ff_lstm_rnn_block,
                   seq_dot_block, seq_sce_block])
    logger = get_logger('ukr_char_lstm_train.log')
    # learning_rate_policy = FixedValuePolicy(0.0005)
    learning_rate_policy = FixedValuePolicy(0.000001)
    # momentum_policy = ScheduledValuePolicy({0: 0.9}, 'momentum', logger)
    momentum_policy = ScheduledValuePolicy({0: 0.99}, 'momentum', logger)
    saver = Hdf5Saver(p.parameters, 200, 'ukr_char_lstm.hdf5', logger)
    criterion = MaxIterCriterion(5000000000)
    sgd_step = SparseSgdStep([p['embd_W']], learning_rate_policy)
    nag_params = dict(p.trainable_parameters)
    del nag_params['embd_W']
    nag_step = NagStep(nag_params.values(), learning_rate_policy, momentum_policy)
    data_block.blocking_contexts = nag_step.blocking_contexts + sgd_step.blocking_contexts
    train_loss_tracker = TrainLossTracker(model, 25, logger)

    class DeppendSetter(object):
        def notify(self):
            data_block.blocking_contexts = nag_step.blocking_contexts + sgd_step.blocking_contexts

    optimizer = Optimizer(criterion, model)
    optimizer.add_observer(momentum_policy)
    optimizer.add_observer(sgd_step)
    optimizer.add_observer(nag_step)
    optimizer.add_observer(DeppendSetter())
    optimizer.add_observer(train_loss_tracker)
    optimizer.add_observer(saver)
    optimizer.add_observer(criterion)
    optimizer.optimize()

    for device_id in xrange(cudart.cuda_get_device_count()):
        cudart.cuda_set_device(device_id)
        cudart.cuda_device_synchronize()