import quagga
import theano
import numpy as np
from itertools import izip
from unittest import TestCase
from theano import tensor as T
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import DotBlock
from quagga.blocks import LstmBlock
from quagga.matrix import MatrixList
from quagga.connector import Connector
from quagga.blocks import SoftmaxCeBlock
from quagga.blocks import SigmoidCeBlock
from quagga.blocks import SequencerBlock


class TestSequentialLstmBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 10

    @classmethod
    def get_orthogonal_matrix(cls, nrows, ncols):
        shape = (nrows, ncols)
        a = cls.rng.normal(0.0, 1.0, shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == shape else v
        q = q.reshape(shape).astype(np.float32)
        return q

    def test_fprop(self):
        """
        compare `fprop` results for cpu and gpu backends
        """

        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            batch_size = self.rng.random_integers(256)
            input_dim, hidden_dim = self.rng.random_integers(1500, size=2)
            x = [self.rng.randn(batch_size, input_dim).astype(np.float32) for _ in xrange(max_input_sequence_len)]
            mask = (self.rng.rand(batch_size, sequence_len) < 0.8).astype(np.float32)
            h_0 = self.rng.randn(batch_size, hidden_dim).astype(np.float32)
            c_0 = self.rng.randn(batch_size, hidden_dim).astype(np.float32)
            W_z = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W_i = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W_f = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W_o = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W = np.hstack((W_z, W_i, W_f, W_o))
            R_z = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R_i = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R_f = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R_o = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R = np.hstack((R_z, R_i, R_f, R_o))

            qh = {}
            for reverse in [False, True]:
                for with_mask in [False, True]:
                    for processor_type in ['gpu', 'cpu']:
                        quagga.processor_type = processor_type
                        context = Context()
                        qx = MatrixList([Connector(Matrix.from_npa(e)) for e in x])
                        qmask = Connector(Matrix.empty(batch_size, qx.length, 'float'))
                        qh_0 = Connector(Matrix.from_npa(h_0))
                        qc_0 = Connector(Matrix.from_npa(c_0))
                        qW = Connector(Matrix.from_npa(W))
                        qR = Connector(Matrix.from_npa(R))
                        lstm = SequencerBlock(LstmBlock, [qW, qR], [qx], ['h'], ['c', 'h'], [qc_0, qh_0], reverse, mask=qmask if with_mask else None)

                        qx.length = sequence_len
                        for e in qx:
                            e.fprop()
                        qmask.assign_npa(context, mask)
                        qmask.fprop()
                        qh_0.fprop()
                        qc_0.fprop()
                        qW.fprop()
                        qR.fprop()
                        lstm.fprop()
                        qh[processor_type] = lstm.h.to_host()

                    for h_gpu, h_cpu in izip(qh['gpu'], qh['cpu']):
                        if not np.allclose(h_gpu, h_cpu, rtol=1e-7, atol=1e-3):
                            r.append(False)
                            break
                    else:
                        r.append(True)

        self.assertEqual(sum(r), len(r))

    def test_bprop(self):
        """
        compare `bprop` results for cpu and gpu backends
        """

        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            batch_size = self.rng.random_integers(256)
            input_dim, hidden_dim = self.rng.random_integers(1500, size=2)

            x = [self.rng.randn(batch_size, input_dim).astype(np.float32) for _ in xrange(max_input_sequence_len)]
            true_labels = [self.rng.randint(2, size=(batch_size, 1)).astype(np.float32) for _ in xrange(max_input_sequence_len)]
            mask = (self.rng.rand(batch_size, sequence_len) < 0.8).astype(np.float32)
            h_0 = self.rng.randn(batch_size, hidden_dim).astype(np.float32)
            c_0 = self.rng.randn(batch_size, hidden_dim).astype(np.float32)
            W_z = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W_i = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W_f = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W_o = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W = np.hstack((W_z, W_i, W_f, W_o))
            R_z = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R_i = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R_f = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R_o = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R = np.hstack((R_z, R_i, R_f, R_o))
            lr_W = self.get_orthogonal_matrix(hidden_dim, 1)
            lr_b = self.rng.rand(1, 1).astype(dtype=np.float32)
            device_id = 0

            quagga_grads = {}
            for reverse in [False, True]:
                for with_mask in [False, True]:
                    for learn_inital_states in [False, True]:
                        for processor_type in ['gpu', 'cpu']:
                            quagga.processor_type = processor_type
                            context = Context()
                            qx = MatrixList([Connector(Matrix.from_npa(e), device_id) for e in x])
                            qtrue_labels = MatrixList([Connector(Matrix.from_npa(e)) for e in true_labels], qx.length)
                            qmask = Connector(Matrix.empty(batch_size, qx.length, 'float'))
                            qh_0 = Connector(Matrix.from_npa(h_0), device_id if learn_inital_states else None)
                            qc_0 = Connector(Matrix.from_npa(c_0), device_id if learn_inital_states else None)
                            qW = Connector(Matrix.from_npa(W), device_id)
                            qR = Connector(Matrix.from_npa(R), device_id)
                            qlr_W = Connector(Matrix.from_npa(lr_W), device_id)
                            qlr_b = Connector(Matrix.from_npa(lr_b), device_id)
                            lstm = SequencerBlock(block_class=LstmBlock,
                                                  params=[qW, qR],
                                                  sequences=[qx],
                                                  output_names=['h'],
                                                  prev_names=['c', 'h'],
                                                  paddings=[qc_0, qh_0],
                                                  reverse=reverse,
                                                  mask=qmask if with_mask else None)
                            seq_dot_block = SequencerBlock(block_class=DotBlock,
                                                           params=[qlr_W, qlr_b],
                                                           sequences=[lstm.h],
                                                           output_names=['output'])
                            seq_sce_block = SequencerBlock(block_class=SigmoidCeBlock,
                                                           params=[],
                                                           sequences=[seq_dot_block.output, qtrue_labels],
                                                           mask=qmask if with_mask else None)
                            qx.length = sequence_len
                            for e in qx:
                                e.fprop()
                            for e in qtrue_labels:
                                e.fprop()
                            qmask.assign_npa(context, mask)
                            qmask.fprop()
                            qlr_W.fprop()
                            qlr_b.fprop()
                            qh_0.fprop()
                            qc_0.fprop()
                            qW.fprop()
                            qR.fprop()
                            lstm.fprop()
                            seq_dot_block.fprop()
                            seq_sce_block.fprop()
                            seq_sce_block.bprop()
                            seq_dot_block.bprop()
                            lstm.bprop()
                            quagga_grads[processor_type] = [qlr_b.backward_matrix.to_host(),
                                                            qlr_W.backward_matrix.to_host(),
                                                            qW.backward_matrix.to_host(),
                                                            qR.backward_matrix.to_host()]
                            if learn_inital_states:
                                quagga_grads[processor_type].append(qc_0.backward_matrix.to_host())
                                quagga_grads[processor_type].append(qh_0.backward_matrix.to_host())
                            quagga_grads[processor_type].extend([e.backward_matrix.to_host() for e in qx])

                        for grad_gpu, grad_cpu in izip(quagga_grads['gpu'], quagga_grads['cpu']):
                            r.append(np.allclose(grad_gpu, grad_cpu, atol=1e-6))

        self.assertEqual(sum(r), len(r))

    def test_theano_fprop(self):
        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            batch_size = self.rng.random_integers(256)
            input_dim, hidden_dim = self.rng.random_integers(1500, size=2)
            x = [self.rng.randn(batch_size, input_dim).astype(np.float32) for _ in xrange(max_input_sequence_len)]
            mask = (self.rng.rand(batch_size, sequence_len) < 0.8).astype(np.float32)
            h_0 = self.rng.randn(batch_size, hidden_dim).astype(np.float32)
            c_0 = self.rng.randn(batch_size, hidden_dim).astype(np.float32)
            W_z = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W_i = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W_f = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W_o = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W = np.hstack((W_z, W_i, W_f, W_o))
            R_z = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R_i = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R_f = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R_o = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R = np.hstack((R_z, R_i, R_f, R_o))

            for reverse in [False, True]:
                for with_mask in [False, True]:
                    context = Context()
                    qx = MatrixList([Connector(Matrix.from_npa(e)) for e in x])
                    qmask = Connector(Matrix.empty(batch_size, qx.length, 'float'))
                    qh_0 = Connector(Matrix.from_npa(h_0))
                    qc_0 = Connector(Matrix.from_npa(c_0))
                    qW = Connector(Matrix.from_npa(W))
                    qR = Connector(Matrix.from_npa(R))
                    lstm = SequencerBlock(LstmBlock, [qW, qR], [qx], ['h'], ['c', 'h'], [qc_0, qh_0], reverse, mask=qmask if with_mask else None)

                    qx.length = sequence_len
                    for e in qx:
                        e.fprop()
                    qmask.assign_npa(context, mask)
                    qmask.fprop()
                    qh_0.fprop()
                    qc_0.fprop()
                    qW.fprop()
                    qR.fprop()
                    lstm.fprop()
                    q_h = lstm.h.to_host()

                    th_x = T.ftensor3()
                    lstm_layer = LstmLayer(W, R, c_0, h_0, reverse)
                    if with_mask:
                        th_mask = T.fmatrix()
                        get_th_h = theano.function([th_x, th_mask], lstm_layer.get_output_expr(th_x, th_mask))
                        th_h = get_th_h(np.dstack(x[:sequence_len]), mask[:, :sequence_len])
                    else:
                        get_th_h = theano.function([th_x], lstm_layer.get_output_expr(th_x))
                        th_h = get_th_h(np.dstack(x[:sequence_len]))

                    for i in xrange(th_h.shape[0]):
                        if not np.allclose(q_h[i], th_h[i]):
                            r.append(False)
                            break
                    else:
                        r.append(True)

        self.assertEqual(sum(r), len(r))

    def test_theano_grad(self):
        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            batch_size = self.rng.random_integers(128)
            input_dim, hidden_dim, class_num = self.rng.random_integers(1500, size=3)

            x = [self.rng.randn(batch_size, input_dim).astype(np.float32) for _ in xrange(max_input_sequence_len)]
            true_labels = [self.rng.randint(class_num, size=(batch_size, 1)).astype(np.int32) for _ in xrange(max_input_sequence_len)]
            mask = (self.rng.rand(batch_size, sequence_len) < 0.8).astype(np.float32)
            h_0 = self.rng.randn(batch_size, hidden_dim).astype(np.float32)
            c_0 = self.rng.randn(batch_size, hidden_dim).astype(np.float32)
            W_z = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W_i = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W_f = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W_o = self.get_orthogonal_matrix(input_dim, hidden_dim)
            W = np.hstack((W_z, W_i, W_f, W_o))
            R_z = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R_i = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R_f = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R_o = self.get_orthogonal_matrix(hidden_dim, hidden_dim)
            R = np.hstack((R_z, R_i, R_f, R_o))
            lr_W = self.get_orthogonal_matrix(hidden_dim, class_num)
            lr_b = self.rng.rand(1, class_num).astype(dtype=np.float32)
            device_id = 0

            for reverse in [False, True]:
                for with_mask in [False, True]:
                    for learn_inital_states in [False, True]:
                        # quagga model
                        context = Context()
                        qx = MatrixList([Connector(Matrix.from_npa(e), device_id) for e in x])
                        qtrue_labels = MatrixList([Connector(Matrix.from_npa(e)) for e in true_labels], qx.length)
                        qmask = Connector(Matrix.empty(batch_size, qx.length, 'float'))
                        qh_0 = Connector(Matrix.from_npa(h_0), device_id if learn_inital_states else None)
                        qc_0 = Connector(Matrix.from_npa(c_0), device_id if learn_inital_states else None)
                        qW = Connector(Matrix.from_npa(W), device_id)
                        qR = Connector(Matrix.from_npa(R), device_id)
                        qlr_W = Connector(Matrix.from_npa(lr_W), device_id)
                        qlr_b = Connector(Matrix.from_npa(lr_b), device_id)
                        lstm = SequencerBlock(block_class=LstmBlock,
                                              params=[qW, qR],
                                              sequences=[qx],
                                              output_names=['h'],
                                              prev_names=['c', 'h'],
                                              paddings=[qc_0, qh_0],
                                              reverse=reverse,
                                              mask=qmask if with_mask else None)
                        seq_dot_block = SequencerBlock(block_class=DotBlock,
                                                       params=[qlr_W, qlr_b],
                                                       sequences=[lstm.h],
                                                       output_names=['output'])
                        seq_sce_block = SequencerBlock(block_class=SoftmaxCeBlock,
                                                       params=[],
                                                       sequences=[seq_dot_block.output, qtrue_labels],
                                                       mask=qmask if with_mask else None)
                        qx.length = sequence_len
                        for e in qx:
                            e.fprop()
                        for e in qtrue_labels:
                            e.fprop()
                        qmask.assign_npa(context, mask)
                        qmask.fprop()
                        qlr_W.fprop()
                        qlr_b.fprop()
                        qh_0.fprop()
                        qc_0.fprop()
                        qW.fprop()
                        qR.fprop()
                        lstm.fprop()
                        seq_dot_block.fprop()
                        seq_sce_block.fprop()
                        seq_sce_block.bprop()
                        seq_dot_block.bprop()
                        lstm.bprop()
                        quagga_grads = [qlr_b.backward_matrix.to_host(),
                                        qlr_W.backward_matrix.to_host(),
                                        qW.backward_matrix.to_host(),
                                        qR.backward_matrix.to_host()]
                        if learn_inital_states:
                            quagga_grads.append(qc_0.backward_matrix.to_host())
                            quagga_grads.append(qh_0.backward_matrix.to_host())
                        quagga_grads.append([e.backward_matrix.to_host() for e in qx])
                        del qx
                        del qlr_b
                        del qlr_W
                        del qW
                        del qR
                        del qmask
                        del lstm
                        del seq_dot_block
                        del seq_sce_block

                        # theano model
                        th_x = T.ftensor3()
                        th_true_labels = T.imatrix()
                        th_mask = T.fmatrix()
                        lstm_layer = LstmLayer(W, R, c_0, h_0, reverse=reverse)
                        th_h = lstm_layer.get_output_expr(th_x, th_mask if with_mask else None)
                        seq_softmax_layer = SequentialSoftmaxLayer(lr_W, lr_b, reverse)
                        loss = seq_softmax_layer.get_loss(th_h, th_true_labels, th_mask if with_mask else None)
                        wrt = [seq_softmax_layer.b, seq_softmax_layer.W, lstm_layer.W, lstm_layer.R]
                        if learn_inital_states:
                            wrt.append(lstm_layer.c0)
                            wrt.append(lstm_layer.h0)
                        wrt.append(th_x)
                        grads = T.grad(loss, wrt)
                        if with_mask:
                            get_theano_grads = theano.function([th_x, th_true_labels, th_mask], grads)
                            theano_grads = get_theano_grads(np.dstack(x[:sequence_len]), np.hstack(true_labels[:sequence_len]), mask[:, :sequence_len])
                        else:
                            get_theano_grads = theano.function([th_x, th_true_labels], grads)
                            theano_grads = get_theano_grads(np.dstack(x[:sequence_len]), np.hstack(true_labels[:sequence_len]))

                        for quagga_grad, theano_grad in izip(quagga_grads[:-1], theano_grads[:-1]):
                            r.append(np.allclose(quagga_grad, theano_grad, atol=1e-6))
                        for i in xrange(theano_grads[-1].shape[-1]):
                            if not np.allclose(quagga_grads[-1][i], theano_grads[-1][..., i], atol=1e-6):
                                r.append(False)
                                break
                        else:
                            r.append(True)

        self.assertEqual(sum(r), len(r))


class LstmLayer(object):
    def __init__(self, W, R, c0, h0, reverse):
        self.W = theano.shared(W, name='W_zifo')
        self.R = theano.shared(R, name='R_zifo')
        self.c0 = theano.shared(c0, name='c0')
        self.h0 = theano.shared(h0, name='h0')
        self.n = W.shape[1] / 4
        self.reverse = reverse

    def get_output_expr(self, input_sequence, mask=None):
        input_sequence = input_sequence.transpose(2, 0, 1)
        mask = mask.T if mask else T.ones(input_sequence.shape[:2], dtype=np.float32)

        if self.reverse:
            input_sequence = input_sequence[::-1]
            mask = mask[::-1]
        [_, h], _ = theano.scan(fn=self.__get_lstm_mask_step,
                                sequences=[input_sequence, mask],
                                outputs_info=[self.c0, self.h0])
        return h[::-1] if self.reverse else h

    def __get_lstm_mask_step(self, x_t, mask_t, c_tm1, h_tm1):
        sigm = T.nnet.sigmoid
        tanh = T.tanh
        dot = theano.dot

        zifo_t = dot(x_t, self.W) + dot(h_tm1, self.R)
        z_t = tanh(zifo_t[:, 0*self.n:1*self.n])
        i_t = sigm(zifo_t[:, 1*self.n:2*self.n])
        f_t = sigm(zifo_t[:, 2*self.n:3*self.n])
        o_t = sigm(zifo_t[:, 3*self.n:4*self.n])

        mask_t = mask_t.dimshuffle(0, 'x')
        c_t = (i_t * z_t + f_t * c_tm1) * mask_t
        h_t = o_t * tanh(c_t) * mask_t

        return mask_t * c_t + (1.0 - mask_t) * c_tm1, \
               mask_t * h_t + (1.0 - mask_t) * h_tm1


class SequentialSoftmaxLayer(object):
    def __init__(self, W, b, reverse):
        self.W = theano.shared(W)
        self.b = theano.shared(b, broadcastable=(True, False))
        self.reverse = reverse

    def get_loss(self, input_sequence, true_labels, mask=None):
        input_sequence = T.dot(input_sequence, self.W) + self.b
        true_labels = true_labels.T
        mask = mask.T if mask else T.ones(input_sequence.shape[:2], np.float32)
        losses, _ = theano.scan(fn=self.__step_loss,
                                sequences=[input_sequence, mask, true_labels])
        return T.sum(losses)

    def __step_loss(self, x_t, mask_t, true_labels_t):
        probs = T.nnet.softmax(x_t)
        return T.mean(mask_t * T.nnet.categorical_crossentropy(probs, true_labels_t))