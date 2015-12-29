# -*- coding: utf-8 -*-
import glob
import h5py
import cPickle
import numpy as np
from quagga import Model
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import DotBlock
from quagga.blocks import LstmBlock
from numpy.random import RandomState
from quagga.blocks import SoftmaxBlock
from quagga.connector import Connector
from quagga.blocks import RowSlicingBlock
from quagga.blocks import ParameterContainer
from quagga.utils.initializers import H5pyInitializer


class DataBlock(object):
    def __init__(self, word_to_idx, device_id):
        self.context = Context(device_id)
        device_id = self.context.device_id
        self.word_idx = Connector(Matrix.empty(1, 1, 'int', device_id))
        self.word_to_idx = word_to_idx
        self.word = None

    def fprop(self):
        word_npa = np.zeros((1, 1), np.int32, 'F')
        word_npa[0][0] = self.word_to_idx[self.word] if self.word in self.word_to_idx else self.word_to_idx['<UNK>']
        self.word_idx.assign_npa(self.context, word_npa)
        self.word_idx.fprop()


with open('/home/sergii/Desktop/quagga/examples/autoencoder/vocab.pckl') as f:
    vocab = cPickle.load(f)
word_to_idx = vocab['word_to_idx']
idx_to_word = vocab['idx_to_word']
# model_file_name = '2auto.hdf5'
# model_file_name = '0_75drop_auto.hdf5'
# model_file_name = '0_95_drop_auto.hdf5'
model_file_name = '/home/sergii/Desktop/quagga/examples/autoencoder/drop_auto.hdf5'
enc_lstm_c0 = H5pyInitializer(model_file_name, 'enc_lstm_c0')()
enc_lstm_h0 = H5pyInitializer(model_file_name, 'enc_lstm_h0')()
p = ParameterContainer(embd_W={'init': H5pyInitializer(model_file_name, 'embd_W'),
                               'device_id': 1,
                               'trainable': False},
                       enc_lstm_c0={'init': lambda: enc_lstm_c0,
                                    'device_id': 1,
                                    'trainable': False},
                       enc_lstm_h0={'init': lambda: enc_lstm_h0,
                                    'device_id': 1,
                                    'trainable': False},
                       enc_lstm_W={'init': H5pyInitializer(model_file_name, 'enc_lstm_W'),
                                   'device_id': 1,
                                   'trainable': False},
                       enc_lstm_R={'init': H5pyInitializer(model_file_name, 'enc_lstm_R'),
                                   'device_id': 1,
                                   'trainable': False})
data_block = DataBlock(word_to_idx, device_id=1)
enc_embd_block = RowSlicingBlock(p['embd_W'], data_block.word_idx)
enc_lstm_block = LstmBlock(p['enc_lstm_W'], p['enc_lstm_R'], None, enc_embd_block.output, None, p['enc_lstm_c0'], p['enc_lstm_h0'], device_id=1)
encoder_model = Model([p, data_block, enc_embd_block, enc_lstm_block])


def encoder_step(word, begin=False):
    data_block.word = word
    if begin:
        enc_lstm_block.prev_c.assign_npa(enc_lstm_block.f_context, enc_lstm_c0)
        enc_lstm_block.prev_h.assign_npa(enc_lstm_block.f_context, enc_lstm_h0)
    else:
        enc_lstm_block.prev_c.assign(enc_lstm_block.f_context, enc_lstm_block.c)
        enc_lstm_block.prev_h.assign(enc_lstm_block.f_context, enc_lstm_block.h)
    encoder_model.fprop()
    data_block.context.wait(enc_lstm_block.f_context)


def get_code(datum):
    for i, e in enumerate(datum):
        encoder_step(e, not i)
    return enc_lstm_block.h.to_host()


dec_lstm_c0 = H5pyInitializer(model_file_name, 'dec_lstm_c0')()
p = ParameterContainer(embd_W={'init': H5pyInitializer(model_file_name, 'embd_W'),
                               'device_id': 1,
                               'trainable': False},
                       dec_lstm_c0={'init': lambda: dec_lstm_c0,
                                    'device_id': 1,
                                    'trainable': False},
                       dec_lstm_W={'init': H5pyInitializer(model_file_name, 'dec_lstm_W'),
                                   'device_id': 1,
                                   'trainable': False},
                       dec_lstm_R={'init': H5pyInitializer(model_file_name, 'dec_lstm_R'),
                                   'device_id': 1,
                                   'trainable': False},
                       sce_dot_block_W={'init': H5pyInitializer(model_file_name, 'sce_dot_block_W'),
                                        'device_id': 1,
                                        'trainable': False},
                       sce_dot_block_b={'init': H5pyInitializer(model_file_name, 'sce_dot_block_b'),
                                        'device_id': 1,
                                        'trainable': False})
dec_embd_block = RowSlicingBlock(p['embd_W'], data_block.word_idx)
dec_lstm_block = LstmBlock(p['dec_lstm_W'], p['dec_lstm_R'], None, dec_embd_block.output, None, p['dec_lstm_c0'], enc_lstm_block.h, device_id=1)
dot_block = DotBlock(p['sce_dot_block_W'], p['sce_dot_block_b'], dec_lstm_block.h, device_id=1)
sce_block = SoftmaxBlock(dot_block.output, device_id=1)
decoder_model = Model([p, data_block, dec_embd_block,
                       dec_lstm_block, dot_block, sce_block])


def decoder_step(word, begin=False):
    data_block.word = word
    if begin:
        dec_lstm_block.prev_c.assign_npa(dec_lstm_block.f_context, dec_lstm_c0)
        dec_lstm_block.prev_h.assign(dec_lstm_block.f_context, enc_lstm_block.h)
    else:
        dec_lstm_block.prev_c.assign(dec_lstm_block.f_context, dec_lstm_block.c)
        dec_lstm_block.prev_h.assign(dec_lstm_block.f_context, dec_lstm_block.h)
    decoder_model.fprop()


def decode_code(code):
    enc_lstm_block.h.assign_npa(enc_lstm_block.f_context, code)
    sentence = []
    word = '<<S>>'
    while True:
        decoder_step(word, word == '<<S>>')
        probs = sce_block.probs.to_host()[0]
        word_idx = np.argmax(probs)
        sentence.append(idx_to_word[word_idx])
        word = sentence[-1]
        if word == '<<S>>':
            break
        word = '<UNK>'
    return sentence


def load_dataset(max_sent_len=42):
    valid_data = []
    for file_path in glob.glob('data/valid/*'):
        print file_path
        with open(file_path) as f:
            for line in f:
                line = line.decode('utf-8').split()
                if len(line) < max_sent_len:
                    valid_data.append(line)
    RandomState(42).shuffle(valid_data)
    valid_data = valid_data[:6000]
    return valid_data


def get_validation_data_codes():
    data = load_dataset()
    print 'ggggo'
    codes = [get_code(datum)[0] for datum in data]
    # with h5py.File('075drop_codes.hdf5', 'w') as h5_file:
    with h5py.File('red_drop_codes.hdf5', 'w') as h5_file:
        h5_file['codes'] = np.vstack(codes)


def test():
    pass
    # data = load_dataset()
    # e = data[1000]

    # ========================================================================

    # e1 = 'I am feeling not very great today .'.split()
    # e2 = 'I feel kind of not bad .'.split()
    # e3 = 'I do not feel well .'.split()
    # e4 = 'I am feeling well now .'.split()
    # e5 = 'I feel blue .'.split()
    # e6 = 'I am under the weather .'.split()
    #
    # code1 = get_code(e1)[0]
    # code2 = get_code(e2)[0]
    # code3 = get_code(e3)[0]
    # code4 = get_code(e4)[0]
    # code5 = get_code(e5)[0]
    # code6 = get_code(e6)[0]
    #
    # norm1 = np.linalg.norm(code1)
    # norm2 = np.linalg.norm(code2)
    # norm3 = np.linalg.norm(code3)
    # norm4 = np.linalg.norm(code4)
    # norm5 = np.linalg.norm(code5)
    # norm6 = np.linalg.norm(code6)
    #
    # print ' '.join(e1)
    # print np.dot(code1, code2) / (norm1 * norm2), ' '.join(e2)
    # print np.dot(code1, code3) / (norm1 * norm3), ' '.join(e3)
    # print np.dot(code1, code4) / (norm1 * norm4), ' '.join(e4)
    # print np.dot(code1, code5) / (norm1 * norm5), ' '.join(e5)
    # print np.dot(code1, code6) / (norm1 * norm6), ' '.join(e6)

    # ========================================================================

    # e = 'This should be very simple example of this algorithm because it can remember a very long sentences even like this one .'.split()
    # e = 'I know how to deal with this incredible nature of such a complex world where we are living .'.split()
    # print len(e)
    # code = get_code(e)
    # print code
    # e = decode_code(code)
    # print len(e) - 1
    # print ' '.join(e)

    # ========================================================================

    # code = np.random.randn(1, 1024).astype(np.float32)
    # e = decode_code(code)
    # print len(e) - 1
    # print ' '.join(e)

    # ========================================================================

    # code1 = get_code('This should be a very simple example of this algorithm because it can remember very long sentences even like this one .'.split())
    # code2 = get_code('I know how to deal with this complexity .'.split())
    # codes = []
    # for t in np.linspace(0, 1, num=30):
    #     codes.append(code1 * (1.0 - t) + t * code2)
    # for c in codes:
    #     print ' '.join(decode_code(c))

if __name__ == '__main__':
    # test()
    get_validation_data_codes()