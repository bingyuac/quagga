# -*- coding: utf-8 -*-
import cPickle
import numpy as np
from quagga import Model
from quagga.matrix import Matrix
from scipy.spatial import distance
from quagga.context import Context
from quagga.blocks import DotBlock
from quagga.blocks import LstmBlock
from quagga.connector import Connector
from quagga.blocks import SoftmaxBlock
from quagga.blocks import RowSlicingBlock
from quagga.blocks import ParameterContainer
from quagga.utils.initializers import H5pyInitializer


class DataBlock(object):
    def __init__(self, char_to_idx, device_id):
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.char_idx = Connector(Matrix.empty(1, 1, 'int', device_id))
        self.char_to_idx = char_to_idx
        self.char = None

    def fprop(self):
        char_npa = np.zeros((1, 1), np.int32, 'F')
        char_npa[0][0] = self.char_to_idx[self.char] if self.char in self.char_to_idx else self.char_to_idx['<unk>']
        self.char_idx.assign_npa(self.context, char_npa)
        self.char_idx.fprop()


with open('vocab.pckl') as f:
    vocab = cPickle.load(f)
char_to_idx = vocab['char_to_idx']
idx_to_char = vocab['idx_to_char']


model_file_name = 'best_best_ukr_char_lstm.hdf5'
embd_W = H5pyInitializer(model_file_name, 'embd_W')()
f_lstm_c0 = H5pyInitializer(model_file_name, 'f_lstm_c0')()
f_lstm_h0 = H5pyInitializer(model_file_name, 'f_lstm_h0')()
f_lstm_W = H5pyInitializer(model_file_name, 'f_lstm_W')()
f_lstm_R = H5pyInitializer(model_file_name, 'f_lstm_R')()
s_lstm_c0 = H5pyInitializer(model_file_name, 's_lstm_c0')()
s_lstm_h0 = H5pyInitializer(model_file_name, 's_lstm_h0')()
s_lstm_W = H5pyInitializer(model_file_name, 's_lstm_W')()
s_lstm_R = H5pyInitializer(model_file_name, 's_lstm_R')()
t_lstm_c0 = H5pyInitializer(model_file_name, 't_lstm_c0')()
t_lstm_h0 = H5pyInitializer(model_file_name, 't_lstm_h0')()
t_lstm_W = H5pyInitializer(model_file_name, 't_lstm_W')()
t_lstm_R = H5pyInitializer(model_file_name, 't_lstm_R')()
ft_lstm_c0 = H5pyInitializer(model_file_name, 'ft_lstm_c0')()
ft_lstm_h0 = H5pyInitializer(model_file_name, 'ft_lstm_h0')()
ft_lstm_W = H5pyInitializer(model_file_name, 'ft_lstm_W')()
ft_lstm_R = H5pyInitializer(model_file_name, 'ff_lstm_R')()
ff_lstm_c0 = H5pyInitializer(model_file_name, 'ff_lstm_c0')()
ff_lstm_h0 = H5pyInitializer(model_file_name, 'ff_lstm_h0')()
ff_lstm_W = H5pyInitializer(model_file_name, 'ff_lstm_W')()
ff_lstm_R = H5pyInitializer(model_file_name, 'ff_lstm_R')()
dot_block_W = H5pyInitializer(model_file_name, 'sce_dot_block_W')()
dot_block_b = H5pyInitializer(model_file_name, 'sce_dot_block_b')()
p = ParameterContainer(embd_W={'init': lambda: embd_W,
                               'device_id': 1},
                       f_lstm_c0={'init': lambda: f_lstm_c0,
                                  'device_id': 1},
                       f_lstm_h0={'init': lambda: f_lstm_h0,
                                  'device_id': 1},
                       f_lstm_W={'init': lambda: f_lstm_W,
                                 'device_id': 1},
                       f_lstm_R={'init': lambda: f_lstm_R,
                                 'device_id': 1},
                       s_lstm_c0={'init': lambda: s_lstm_c0,
                                  'device_id': 1},
                       s_lstm_h0={'init': lambda: s_lstm_h0,
                                  'device_id': 1},
                       s_lstm_W={'init': lambda: s_lstm_W,
                                 'device_id': 1},
                       s_lstm_R={'init': lambda: s_lstm_R,
                                 'device_id': 1},
                       t_lstm_c0={'init': lambda: t_lstm_c0,
                                  'device_id': 1},
                       t_lstm_h0={'init': lambda: t_lstm_h0,
                                  'device_id': 1},
                       t_lstm_W={'init': lambda: t_lstm_W,
                                 'device_id': 1},
                       t_lstm_R={'init': lambda: t_lstm_R,
                                 'device_id': 1},
                       ft_lstm_c0={'init': lambda: ft_lstm_c0,
                                   'device_id': 1},
                       ft_lstm_h0={'init': lambda: ft_lstm_h0,
                                   'device_id': 1},
                       ft_lstm_W={'init': lambda: ft_lstm_W,
                                  'device_id': 1},
                       ft_lstm_R={'init': lambda: ft_lstm_R,
                                  'device_id': 1},
                       ff_lstm_c0={'init': lambda: ff_lstm_c0,
                                   'device_id': 1},
                       ff_lstm_h0={'init': lambda: ff_lstm_h0,
                                   'device_id': 1},
                       ff_lstm_W={'init': lambda: ff_lstm_W,
                                  'device_id': 1},
                       ff_lstm_R={'init': lambda: ff_lstm_R,
                                  'device_id': 1},
                       sce_dot_block_W={'init': lambda: dot_block_W,
                                        'device_id': 1},
                       sce_dot_block_b={'init': lambda: dot_block_b,
                                        'device_id': 1})
data_block = DataBlock(char_to_idx, device_id=1)
embd_block = RowSlicingBlock(W=p['embd_W'], row_indexes=data_block.char_idx)
f_lstm_rnn_block = LstmBlock(p['f_lstm_W'], p['f_lstm_R'], None, embd_block.output, None, p['f_lstm_c0'], p['f_lstm_h0'], device_id=1)
s_lstm_rnn_block = LstmBlock(p['s_lstm_W'], p['s_lstm_R'], None, f_lstm_rnn_block.h, None, p['s_lstm_c0'], p['s_lstm_h0'], device_id=1)
t_lstm_rnn_block = LstmBlock(p['t_lstm_W'], p['t_lstm_R'], None, s_lstm_rnn_block.h, None, p['t_lstm_c0'], p['t_lstm_h0'], device_id=1)
ft_lstm_rnn_block = LstmBlock(p['ft_lstm_W'], p['ft_lstm_R'], None, t_lstm_rnn_block.h, None, p['ft_lstm_c0'], p['ft_lstm_h0'], device_id=1)
ff_lstm_rnn_block = LstmBlock(p['ff_lstm_W'], p['ff_lstm_R'], None, ft_lstm_rnn_block.h, None, p['ff_lstm_c0'], p['ff_lstm_h0'], device_id=1)
dot_block = DotBlock(p['sce_dot_block_W'], p['sce_dot_block_b'], ff_lstm_rnn_block.h, device_id=1)
softmax_block = SoftmaxBlock(dot_block.output, device_id=1)
model = Model([p, data_block, embd_block,
               f_lstm_rnn_block,
               s_lstm_rnn_block,
               t_lstm_rnn_block,
               ft_lstm_rnn_block,
               ff_lstm_rnn_block,
               dot_block, softmax_block])


def step(char, begin=False):
    data_block.char = char
    if begin:
        p['f_lstm_c0'].assign_npa(f_lstm_rnn_block.f_context, f_lstm_c0)
        p['f_lstm_h0'].assign_npa(f_lstm_rnn_block.f_context, f_lstm_h0)
        p['s_lstm_c0'].assign_npa(s_lstm_rnn_block.f_context, s_lstm_c0)
        p['s_lstm_h0'].assign_npa(s_lstm_rnn_block.f_context, s_lstm_h0)
        p['t_lstm_c0'].assign_npa(t_lstm_rnn_block.f_context, t_lstm_c0)
        p['t_lstm_h0'].assign_npa(t_lstm_rnn_block.f_context, t_lstm_h0)
        p['ft_lstm_c0'].assign_npa(ft_lstm_rnn_block.f_context, ft_lstm_c0)
        p['ft_lstm_h0'].assign_npa(ft_lstm_rnn_block.f_context, ft_lstm_h0)
        p['ff_lstm_c0'].assign_npa(ff_lstm_rnn_block.f_context, ff_lstm_c0)
        p['ff_lstm_h0'].assign_npa(ff_lstm_rnn_block.f_context, ff_lstm_h0)
    else:
        f_lstm_rnn_block.prev_c.assign(f_lstm_rnn_block.f_context, f_lstm_rnn_block.c)
        f_lstm_rnn_block.prev_h.assign(f_lstm_rnn_block.f_context, f_lstm_rnn_block.h)
        s_lstm_rnn_block.prev_c.assign(s_lstm_rnn_block.f_context, s_lstm_rnn_block.c)
        s_lstm_rnn_block.prev_h.assign(s_lstm_rnn_block.f_context, s_lstm_rnn_block.h)
        t_lstm_rnn_block.prev_c.assign(t_lstm_rnn_block.f_context, t_lstm_rnn_block.c)
        t_lstm_rnn_block.prev_h.assign(t_lstm_rnn_block.f_context, t_lstm_rnn_block.h)
        ft_lstm_rnn_block.prev_c.assign(ft_lstm_rnn_block.f_context, ft_lstm_rnn_block.c)
        ft_lstm_rnn_block.prev_h.assign(ft_lstm_rnn_block.f_context, ft_lstm_rnn_block.h)
        ff_lstm_rnn_block.prev_c.assign(ff_lstm_rnn_block.f_context, ff_lstm_rnn_block.c)
        ff_lstm_rnn_block.prev_h.assign(ff_lstm_rnn_block.f_context, ff_lstm_rnn_block.h)
    model.fprop()
    return softmax_block.probs.to_host()


def make_prediction(seed, n_steps):
    seed = u''.join([e for e in seed if e in char_to_idx])
    predicted_chars = []
    for i, char in enumerate(seed):
        probs = step(char, i == 0)
    for i in xrange(n_steps):
        if predicted_chars and predicted_chars[-1] == ' ':
            char_idx = np.random.choice(len(idx_to_char), 1, p=probs[0])
            # char_idx = np.argmax(probs[0])
        else:
            # char_idx = np.random.choice(len(idx_to_char), 1, p=probs[0])
            char_idx = np.argmax(probs[0])
        predicted_chars.append(idx_to_char[char_idx])
        probs = step(predicted_chars[-1])
    return predicted_chars


def get_last_hiddens(word):
    for i, char in enumerate(word):
        step(char, i == 0)
    return np.concatenate([f_lstm_rnn_block.h.to_host()[0],
                           s_lstm_rnn_block.h.to_host()[0],
                           t_lstm_rnn_block.h.to_host()[0],
                           ft_lstm_rnn_block.h.to_host()[0],
                           ff_lstm_rnn_block.h.to_host()[0]])


def get_similarity(first_word, second_word):
    first_hidden = get_last_hiddens(first_word)
    second_hidden = get_last_hiddens(second_word)
    return distance.cosine(first_hidden, second_hidden) - 1.0


if __name__ == '__main__':
    # print get_similarity(u'математичний', u'математика')

    seed = u'До Львова завітал'
    tail = make_prediction(seed, 2000)
    print seed
    print u''.join(tail)