import quagga
import theano
import numpy as np
from itertools import izip
from quagga.utils import List
from unittest import TestCase
from theano import tensor as T
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector
from quagga.blocks import SoftmaxCeBlock
from quagga.blocks import SequencerBlock
from quagga.blocks import RowSlicingBlock


class TestRowSlicingBlock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.N = 15

    @classmethod
    def get_orthogonal_matrix(cls, nrows, ncols):
        shape = (nrows, ncols)
        a = cls.rng.normal(0.0, 1.0, shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == shape else v
        q = q.reshape(shape).astype(np.float32)
        return q

    def test_fprop_vector(self):
        """
        compare `fprop` results for cpu and gpu backends
        """
        r = []
        for _ in xrange(self.N):
            embd_dim = self.rng.random_integers(10000)
            batch_size, output_dim = self.rng.random_integers(2000, size=2)
            W = self.get_orthogonal_matrix(embd_dim, output_dim)
            row_idxs = self.rng.randint(embd_dim, size=(batch_size, 1)).astype(np.int32)

            output = {}
            for processor_type in ['gpu', 'cpu']:
                quagga.processor_type = processor_type
                qrow_idxs = Connector(Matrix.from_npa(row_idxs))
                qW = Connector(Matrix.from_npa(W))
                row_slicing_block = RowSlicingBlock(qW, qrow_idxs)
                qW.fprop()
                qrow_idxs.fprop()
                row_slicing_block.fprop()
                output[processor_type] = row_slicing_block.output.to_host()

            r.append(np.allclose(output['gpu'], output['cpu']))

        self.assertEqual(sum(r), len(r))

    def test_fprop_matrix(self):
        """
        compare `fprop` results for cpu and gpu backends
        """
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(300)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            embd_dim = self.rng.random_integers(10000)
            batch_size, output_dim = self.rng.random_integers(2000, size=2)
            W = self.get_orthogonal_matrix(embd_dim, output_dim)
            row_idxs = self.rng.randint(embd_dim, size=(batch_size, max_input_sequence_len)).astype(np.int32)

            output = {}
            for processor_type in ['gpu', 'cpu']:
                quagga.processor_type = processor_type
                qrow_idxs = Connector(Matrix.from_npa(row_idxs))
                qW = Connector(Matrix.from_npa(W))
                row_slicing_block = RowSlicingBlock(qW, qrow_idxs)
                qW.fprop()
                qrow_idxs.ncols = sequence_len
                qrow_idxs.fprop()
                row_slicing_block.fprop()
                output[processor_type] = row_slicing_block.output.to_host()

            for output_gpu, output_cpu in izip(output['gpu'], output['cpu']):
                r.append(np.allclose(output_gpu, output_cpu))

        self.assertEqual(sum(r), len(r))

    def test_bprop_vector(self):
        r = []
        for _ in xrange(self.N):
            embd_dim = self.rng.random_integers(10000)
            batch_size, output_dim = self.rng.random_integers(2000, size=2)
            W = self.get_orthogonal_matrix(embd_dim, output_dim)
            row_idxs = self.rng.randint(embd_dim, size=(batch_size, 1)).astype(np.int32)
            true_labels = self.rng.randint(output_dim, size=(batch_size, 1)).astype(np.int32)
            device_id = 0

            output = {}
            for processor_type in ['gpu', 'cpu']:
                quagga.processor_type = processor_type
                qrow_idxs = Connector(Matrix.from_npa(row_idxs))
                qtrue_labels = Connector(Matrix.from_npa(true_labels))
                qW = Connector(Matrix.from_npa(W), device_id)
                row_slicing_block = RowSlicingBlock(qW, qrow_idxs)
                sce_block = SoftmaxCeBlock(row_slicing_block.output, qtrue_labels)
                qW.fprop()
                qrow_idxs.fprop()
                row_slicing_block.fprop()
                sce_block.fprop()
                sce_block.bprop()
                row_slicing_block.bprop()
                qW.add(Context(), qW.backward_matrix)
                output[processor_type] = qW.to_host()

            r.append(np.allclose(output['gpu'], output['cpu']))

        self.assertEqual(sum(r), len(r))

    def test_bprop_matrix(self):
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            embd_dim = self.rng.random_integers(10000)
            batch_size = self.rng.random_integers(500)
            output_dim = self.rng.random_integers(2000)
            W = self.get_orthogonal_matrix(embd_dim, output_dim)
            row_idxs = self.rng.randint(embd_dim, size=(batch_size, max_input_sequence_len)).astype(np.int32)
            true_labels = [self.rng.randint(output_dim, size=(batch_size, 1)).astype(np.int32) for _ in xrange(max_input_sequence_len)]
            device_id = 0

            output = {}
            for processor_type in ['gpu', 'cpu']:
                quagga.processor_type = processor_type
                qrow_idxs = Connector(Matrix.from_npa(row_idxs))
                qtrue_labels = List([Connector(Matrix.from_npa(e)) for e in true_labels], qrow_idxs.ncols)
                qW = Connector(Matrix.from_npa(W), device_id)
                row_slicing_block = RowSlicingBlock(qW, qrow_idxs)
                seq_sce_block = SequencerBlock(block_class=SoftmaxCeBlock,
                                               params=[],
                                               sequences=[row_slicing_block.output, qtrue_labels])
                qW.fprop()
                qrow_idxs.ncols = sequence_len
                qrow_idxs.fprop()
                row_slicing_block.fprop()
                seq_sce_block.fprop()
                seq_sce_block.bprop()
                row_slicing_block.bprop()
                qW.add(Context(), qW.backward_matrix)
                output[processor_type] = qW.to_host()

            r.append(np.allclose(output['gpu'], output['cpu']))

        self.assertEqual(sum(r), len(r))

    def test_theano_fprop_vector(self):
        r = []
        for _ in xrange(self.N):
            embd_dim = self.rng.random_integers(10000)
            batch_size, output_dim = self.rng.random_integers(2000, size=2)
            W = self.get_orthogonal_matrix(embd_dim, output_dim)
            row_idxs = self.rng.randint(embd_dim, size=(batch_size, 1)).astype(np.int32)

            quagga.processor_type = 'gpu'
            qrow_idxs = Connector(Matrix.from_npa(row_idxs))
            qW = Connector(Matrix.from_npa(W))
            row_slicing_block = RowSlicingBlock(qW, qrow_idxs)
            qW.fprop()
            qrow_idxs.fprop()
            row_slicing_block.fprop()
            q_output = row_slicing_block.output.to_host()

            trow_idxs = T.ivector()
            row_slicing_layer = RowSlicingLayer(W)
            t_output = row_slicing_layer.get_output_expr(trow_idxs)
            t_output = theano.function([trow_idxs], t_output)(row_idxs[:, 0])

            r.append(np.allclose(q_output, t_output))

        self.assertEqual(sum(r), len(r))

    def test_theano_fprop_matrix(self):
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(300)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            embd_dim = self.rng.random_integers(10000)
            batch_size = self.rng.random_integers(500)
            output_dim = self.rng.random_integers(2000)
            W = self.get_orthogonal_matrix(embd_dim, output_dim)
            row_idxs = self.rng.randint(embd_dim, size=(batch_size, max_input_sequence_len)).astype(np.int32)

            quagga.processor_type = 'gpu'
            qrow_idxs = Connector(Matrix.from_npa(row_idxs))
            qW = Connector(Matrix.from_npa(W))
            row_slicing_block = RowSlicingBlock(qW, qrow_idxs)
            qW.fprop()
            qrow_idxs.ncols = sequence_len
            qrow_idxs.fprop()
            row_slicing_block.fprop()
            q_output = row_slicing_block.output.to_host()

            th_row_idxs = T.imatrix()
            row_slicing_layer = RowSlicingLayer(W)
            toutput = row_slicing_layer.get_output_expr(th_row_idxs)
            th_output = theano.function([th_row_idxs], toutput)(row_idxs)

            for i in xrange(sequence_len):
                r.append(np.allclose(q_output[i], th_output[i]))

        self.assertEqual(sum(r), len(r))

    def test_theano_bprop_vector(self):
        r = []
        for _ in xrange(self.N):
            embd_dim = self.rng.random_integers(10000)
            batch_size, output_dim = self.rng.random_integers(2000, size=2)
            W = self.get_orthogonal_matrix(embd_dim, output_dim)
            row_idxs = self.rng.randint(embd_dim, size=(batch_size, 1)).astype(np.int32)
            true_labels = self.rng.randint(output_dim, size=(batch_size, 1)).astype(np.int32)
            device_id = 0

            quagga.processor_type = 'gpu'
            qrow_idxs = Connector(Matrix.from_npa(row_idxs))
            qW = Connector(Matrix.from_npa(W), device_id)
            qtrue_labels = Connector(Matrix.from_npa(true_labels))
            row_slicing_block = RowSlicingBlock(qW, qrow_idxs)
            sce_block = SoftmaxCeBlock(row_slicing_block.output, qtrue_labels)
            qtrue_labels.fprop()
            qW.fprop()
            qrow_idxs.fprop()
            row_slicing_block.fprop()
            sce_block.fprop()
            sce_block.bprop()
            row_slicing_block.bprop()
            qW.add(Context(), qW.backward_matrix)

            th_row_idxs = T.ivector()
            th_true_labels = T.ivector()
            row_slicing_layer = RowSlicingLayer(W)
            toutput = row_slicing_layer.get_output_expr(th_row_idxs)
            loss = SoftmaxLayer.get_loss(toutput, th_true_labels)
            dL_dW = T.grad(loss, row_slicing_layer.W)
            fun = theano.function([th_row_idxs, th_true_labels],
                                  updates=[(row_slicing_layer.W, row_slicing_layer.W + dL_dW)])
            fun(row_idxs[:, 0], true_labels[:, 0])
            r.append(np.allclose(qW.to_host(), row_slicing_layer.W.get_value()))

        self.assertEqual(sum(r), len(r))

    def test_theano_bprop_matrix(self):
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(300)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(2, max_input_sequence_len)
            embd_dim = self.rng.random_integers(10000)
            batch_size = self.rng.random_integers(500)
            output_dim = self.rng.random_integers(2000)
            W = self.get_orthogonal_matrix(embd_dim, output_dim)
            row_idxs = self.rng.randint(embd_dim, size=(batch_size, max_input_sequence_len)).astype(np.int32)
            true_labels = [self.rng.randint(output_dim, size=(batch_size, 1)).astype(np.int32) for _ in xrange(max_input_sequence_len)]
            device_id = 0

            quagga.processor_type = 'gpu'
            qrow_idxs = Connector(Matrix.from_npa(row_idxs))
            qtrue_labels = List([Connector(Matrix.from_npa(e)) for e in true_labels], qrow_idxs.ncols)
            qW = Connector(Matrix.from_npa(W), device_id)
            row_slicing_block = RowSlicingBlock(qW, qrow_idxs)
            seq_sce_block = SequencerBlock(block_class=SoftmaxCeBlock,
                                           params=[],
                                           sequences=[row_slicing_block.output, qtrue_labels])
            qW.fprop()
            qrow_idxs.ncols = sequence_len
            qrow_idxs.fprop()
            row_slicing_block.fprop()
            seq_sce_block.fprop()
            seq_sce_block.bprop()
            row_slicing_block.bprop()
            qW.add(Context(), qW.backward_matrix)

            th_row_idxs = T.imatrix()
            th_true_labels = T.imatrix()
            row_slicing_layer = RowSlicingLayer(W)
            toutput = row_slicing_layer.get_output_expr(th_row_idxs)
            loss = SequentialSoftmaxLayer.get_loss(toutput, th_true_labels)
            dL_dW = T.grad(loss, row_slicing_layer.W)
            fun = theano.function([th_row_idxs, th_true_labels],
                                  updates=[(row_slicing_layer.W, row_slicing_layer.W + dL_dW)])
            fun(row_idxs, np.hstack(true_labels[:sequence_len]))

            r.append(np.allclose(qW.to_host(), row_slicing_layer.W.get_value(), atol=1e-5))

        self.assertEqual(sum(r), len(r))


class RowSlicingLayer(object):
    def __init__(self, W):
        self.W = theano.shared(W)

    def get_output_expr(self, input_sequence):
        if input_sequence.ndim == 2:
            return self.W[input_sequence].transpose(1, 0, 2)
        return self.W[input_sequence]


class SoftmaxLayer(object):
    @staticmethod
    def get_loss(x, true_labels):
        probs = T.nnet.softmax(x)
        return T.mean(T.nnet.categorical_crossentropy(probs, true_labels))


class SequentialSoftmaxLayer(object):
    @staticmethod
    def get_loss(input_sequence, true_labels):
        true_labels = true_labels.T
        losses, _ = theano.scan(fn=SequentialSoftmaxLayer.__step_loss,
                                sequences=[input_sequence, true_labels])
        return T.sum(losses)

    @staticmethod
    def __step_loss(x_t, true_labels_t):
        probs = T.nnet.softmax(x_t)
        return T.mean(T.nnet.categorical_crossentropy(probs, true_labels_t))