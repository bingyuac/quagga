# ----------------------------------------------------------------------------
# Copyright 2015 Grammarly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from itertools import izip
from unittest import TestCase

import theano
import numpy as np
from theano import tensor as T

import quagga
from quagga.utils import List
from quagga.matrix import Matrix
from quagga.blocks import DotBlock
from quagga.connector import Connector
from quagga.blocks import SoftmaxCeBlock
from quagga.blocks import SequencerBlock


class TestSequentialDotBlock(TestCase):
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
            W = self.get_orthogonal_matrix(input_dim, hidden_dim)
            b = self.rng.rand(1, hidden_dim).astype(np.float32)

            from quagga.cuda import cudart
            cudart.cuda_set_device(1)

            qoutput = {}
            for reverse in [False, True]:
                for with_bias in [False, True]:
                    for processor_type in ['gpu', 'cpu']:
                        quagga.processor_type = processor_type
                        qx = List([Connector(Matrix.from_npa(e)) for e in x])
                        qW = Connector(Matrix.from_npa(W))
                        qb = Connector(Matrix.from_npa(b)) if with_bias else None
                        seq_dot_block = SequencerBlock(block_class=DotBlock,
                                                       params=[qW, qb],
                                                       sequences=[qx],
                                                       output_names=['output'],
                                                       reverse=reverse)
                        qx.length = sequence_len
                        qx.fprop()
                        qW.fprop()
                        if qb:
                            qb.fprop()
                        seq_dot_block.fprop()
                        qoutput[processor_type] = seq_dot_block.output.to_host()

                    for output_gpu, output_cpu in izip(qoutput['gpu'], qoutput['cpu']):
                        if not np.allclose(output_gpu, output_cpu, atol=1e-5):
                            r.append(False)
                            break
                    else:
                        r.append(True)

        self.assertEqual(sum(r), len(r))

    def test_bprop(self):
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
            true_labels = [self.rng.randint(hidden_dim, size=(batch_size, 1)).astype(np.int32) for _ in xrange(max_input_sequence_len)]
            W = self.get_orthogonal_matrix(input_dim, hidden_dim)
            b = self.rng.rand(1, hidden_dim).astype(np.float32)
            device_id = 0

            quagga_grads = {}
            for reverse in [False, True]:
                for with_bias in [False, True]:
                    for processor_type in ['gpu', 'cpu']:
                        quagga.processor_type = processor_type
                        qx = List([Connector(Matrix.from_npa(e), device_id) for e in x])
                        qtrue_labels = List([Connector(Matrix.from_npa(e)) for e in true_labels], len(qx))
                        qW = Connector(Matrix.from_npa(W), device_id)
                        qb = Connector(Matrix.from_npa(b), device_id) if with_bias else None
                        seq_dot_block = SequencerBlock(block_class=DotBlock,
                                                       params=[qW, qb],
                                                       sequences=[qx],
                                                       output_names=['output'],
                                                       reverse=reverse)
                        seq_sce_block = SequencerBlock(block_class=SoftmaxCeBlock,
                                                       params=[],
                                                       sequences=[seq_dot_block.output, qtrue_labels],
                                                       reverse=reverse)
                        qx.length = sequence_len
                        qx.fprop()
                        qtrue_labels.fprop()
                        qW.fprop()
                        if qb:
                            qb.fprop()
                        seq_dot_block.fprop()
                        seq_sce_block.fprop()
                        seq_sce_block.bprop()
                        seq_dot_block.bprop()
                        quagga_grads[processor_type] = [qW.backward_matrix.to_host()]
                        if with_bias:
                            quagga_grads[processor_type].append(qb.backward_matrix.to_host())
                        quagga_grads[processor_type].extend(e.backward_matrix.to_host() for e in qx)

                    for grad_gpu, grad_cpu in izip(quagga_grads['gpu'], quagga_grads['cpu']):
                        r.append(np.allclose(grad_gpu, grad_cpu, atol=1e-5))

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
            W = self.get_orthogonal_matrix(input_dim, hidden_dim)
            b = self.rng.rand(1, hidden_dim).astype(np.float32)

            for reverse in [False, True]:
                for with_bias in [False, True]:
                    qx = List([Connector(Matrix.from_npa(e)) for e in x])
                    qW = Connector(Matrix.from_npa(W))
                    qb = Connector(Matrix.from_npa(b)) if with_bias else None
                    seq_dot_block = SequencerBlock(block_class=DotBlock,
                                                   params=[qW, qb],
                                                   sequences=[qx],
                                                   output_names=['output'],
                                                   reverse=reverse)
                    qx.length = sequence_len
                    qx.fprop()
                    qW.fprop()
                    if qb:
                        qb.fprop()
                    seq_dot_block.fprop()
                    qoutput = seq_dot_block.output.to_host()

                    seq_dot_layer = SequentialDotLayer(W, b if with_bias else None, reverse)
                    th_x = T.ftensor3()
                    get_th_output = theano.function([th_x], seq_dot_layer.get_output_expr(th_x))
                    th_output = get_th_output(np.dstack(x[:sequence_len]))

                    for i in xrange(th_output.shape[0]):
                        if not np.allclose(qoutput[i], th_output[i]):
                            r.append(False)
                            break
                    else:
                        r.append(True)

        self.assertEqual(sum(r), len(r))

    def test_theano_bprop(self):
        quagga.processor_type = 'gpu'
        r = []
        for i in xrange(self.N):
            max_input_sequence_len = self.rng.random_integers(500)
            sequence_len = max_input_sequence_len if i == 0 else self.rng.random_integers(max_input_sequence_len)
            batch_size = self.rng.random_integers(256)
            input_dim, hidden_dim = self.rng.random_integers(1500, size=2)
            x = [self.rng.randn(batch_size, input_dim).astype(np.float32) for _ in xrange(max_input_sequence_len)]
            true_labels = [self.rng.randint(hidden_dim, size=(batch_size, 1)).astype(np.int32) for _ in xrange(max_input_sequence_len)]
            W = self.get_orthogonal_matrix(input_dim, hidden_dim)
            b = self.rng.rand(1, hidden_dim).astype(np.float32)
            device_id = 0

            for reverse in [False, True]:
                for with_bias in [False, True]:
                    qx = List([Connector(Matrix.from_npa(e), device_id) for e in x])
                    qtrue_labels = List([Connector(Matrix.from_npa(e)) for e in true_labels], len(qx))
                    qW = Connector(Matrix.from_npa(W), device_id)
                    qb = Connector(Matrix.from_npa(b), device_id) if with_bias else None
                    seq_dot_block = SequencerBlock(block_class=DotBlock,
                                                   params=[qW, qb],
                                                   sequences=[qx],
                                                   output_names=['output'],
                                                   reverse=reverse)
                    seq_sce_block = SequencerBlock(block_class=SoftmaxCeBlock,
                                                   params=[],
                                                   sequences=[seq_dot_block.output, qtrue_labels],
                                                   reverse=reverse)
                    qx.length = sequence_len
                    qx.fprop()
                    qtrue_labels.fprop()
                    qW.fprop()
                    if qb:
                        qb.fprop()
                    seq_dot_block.fprop()
                    seq_sce_block.fprop()
                    seq_sce_block.bprop()
                    seq_dot_block.bprop()
                    quagga_grads = [qW.backward_matrix.to_host()]
                    if with_bias:
                        quagga_grads.append(qb.backward_matrix.to_host())
                    quagga_grads.append([e.backward_matrix.to_host() for e in qx])

                    seq_dot_layer = SequentialDotLayer(W, b if with_bias else None, reverse)
                    seq_sce_layer = SequentialSoftmaxLayer()
                    th_x = T.ftensor3()
                    th_true_labels = T.imatrix()
                    loss = seq_sce_layer.get_loss(seq_dot_layer.get_output_expr(th_x), th_true_labels)
                    wrt = [seq_dot_layer.W]
                    if with_bias:
                        wrt.append(seq_dot_layer.b)
                    wrt.append(th_x)
                    grads = T.grad(loss, wrt)
                    get_theano_grads = theano.function([th_x, th_true_labels], grads)
                    theano_grads = get_theano_grads(np.dstack(x[:sequence_len]), np.hstack(true_labels[:sequence_len]))

                    for quagga_grad, theano_grad in izip(quagga_grads[:-1], theano_grads[:-1]):
                        r.append(np.allclose(quagga_grad, theano_grad, atol=1e-5))
                    for i in xrange(theano_grads[-1].shape[-1]):
                        if not np.allclose(quagga_grads[-1][i], theano_grads[-1][..., i], atol=1e-5):
                            r.append(False)
                            break
                    else:
                        r.append(True)

        self.assertEqual(sum(r), len(r))


class SequentialDotLayer(object):
    def __init__(self, W, b, reverse):
        self.W = theano.shared(W)
        self.b = theano.shared(b, broadcastable=(True, False)) if b is not None else b
        self.reverse = reverse

    def get_output_expr(self, input_sequence):
        input_sequence = input_sequence.transpose(2, 0, 1)
        output = T.dot(input_sequence, self.W)
        if self.b:
            output += self.b
        return output


class SequentialSoftmaxLayer(object):
    def get_loss(self, input_sequence, true_labels):
        true_labels = true_labels.transpose(1, 0)
        losses, _ = theano.scan(fn=self.__step_loss,
                                sequences=[input_sequence, true_labels])
        return T.sum(losses)

    def __step_loss(self, x_t, true_labels_t):
        probs = T.nnet.softmax(x_t)
        return T.mean(T.nnet.categorical_crossentropy(probs, true_labels_t))