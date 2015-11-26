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
import time
from itertools import izip
from collections import OrderedDict

import numpy as np

import layers
from layers import LstmBlock
from quagga.matrix import GpuMatrix, GpuMatrixContext, CpuMatrix, CpuMatrixContext

rng = np.random.RandomState(seed=42)
MatrixClass = {'cpu': CpuMatrix, 'gpu': GpuMatrix}
MatrixContextClass = {'cpu': CpuMatrixContext, 'gpu': GpuMatrixContext}
context = {'cpu': CpuMatrixContext(), 'gpu': GpuMatrixContext()}

shapes = OrderedDict([((256, 1), 10000),
                      ((512, 1), 10000),
                      ((1024, 1), 10000),
                      ((2048, 1), 10000),
                      ((4096, 1), 10000),
                      ((256, 256), 500),
                      ((512, 512), 400),
                      ((1024, 1024), 200),
                      ((2048, 2048), 100),
                      ((4096, 4096), 50)])


def test_scale():
    for shape, N in shapes.iteritems():
        scale_out_time = {'cpu': [], 'gpu': []}
        scale_inplace_time = {'cpu': [], 'gpu': []}

        for i in xrange(N):
            _a = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            alpha = 2 * rng.rand(1)[0] - 1

            for pt in ['cpu', 'gpu']:
                a = MatrixClass[pt].from_npa(_a)
                out = MatrixClass[pt].empty_like(a)

                t = time.clock()
                a.scale(context[pt], alpha, out)
                context[pt].synchronize()
                scale_out_time[pt].append(time.clock() - t)

                t = time.clock()
                a.scale(context[pt], alpha)
                context[pt].synchronize()
                scale_inplace_time[pt].append(time.clock() - t)

        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(scale_out_time['cpu']), 'cpu scale_out')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(scale_out_time['gpu']), 'gpu scale_out')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(scale_inplace_time['cpu']), 'cpu scale_inplace')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(scale_inplace_time['gpu']), 'gpu scale_inplace')
        print 50 * '-'


def test_tanh():
    for shape, N in shapes.iteritems():
        tanh_time = {'cpu': [], 'gpu': []}
        tanh_der_time = {'cpu': [], 'gpu': []}

        for i in xrange(N):
            _a = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)

            for pt in ['cpu', 'gpu']:
                a = MatrixClass[pt].from_npa(_a)
                tanh_matirx = MatrixClass[pt].empty_like(a)
                derivative_matrix = MatrixClass[pt].empty_like(a)

                t = time.clock()
                a.tanh(context[pt], tanh_matirx)
                context[pt].synchronize()
                tanh_time[pt].append(time.clock() - t)

                t = time.clock()
                a.tanh(context[pt], tanh_matirx, derivative_matrix)
                context[pt].synchronize()
                tanh_der_time[pt].append(time.clock() - t)

        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(tanh_time['cpu']), 'cpu tanh')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(tanh_time['gpu']), 'gpu tanh')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(tanh_der_time['cpu']), 'cpu tanh_der')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(tanh_der_time['gpu']), 'gpu tanh_der')
        print 50 * '-'


def test_sigmoid():
    for shape, N in shapes.iteritems():
        sigmoid_time = {'cpu': [], 'gpu': []}
        sigmoid_der_time = {'cpu': [], 'gpu': []}

        for i in xrange(N):
            _a = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)

            for pt in ['cpu', 'gpu']:
                layers.MatrixClass = MatrixClass[pt]
                a = MatrixClass[pt].from_npa(_a)
                sigmoid_matirx = MatrixClass[pt].empty_like(a)
                derivative_matrix = MatrixClass[pt].empty_like(a)

                t = time.clock()
                a.sigmoid(context[pt], sigmoid_matirx)
                context[pt].synchronize()
                sigmoid_time[pt].append(time.clock() - t)

                t = time.clock()
                a.sigmoid(context[pt], sigmoid_matirx, derivative_matrix)
                context[pt].synchronize()
                sigmoid_der_time[pt].append(time.clock() - t)

        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(sigmoid_time['cpu']), 'cpu sigmoid')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(sigmoid_time['gpu']), 'gpu sigmoid')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(sigmoid_der_time['cpu']), 'cpu sigmoid_der')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(sigmoid_der_time['gpu']), 'gpu sigmoid_der')
        print 50 * '-'


def test_add():
    for shape, N in shapes.iteritems():
        add_scaled_time = {'cpu': [], 'gpu': []}
        add_time = {'cpu': [], 'gpu': []}

        for i in xrange(N):
            _a = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _b = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _c = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _d = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            alpha = 2 * rng.rand(1)[0] - 1

            for pt in ['cpu', 'gpu']:
                a = MatrixClass[pt].from_npa(_a)
                b = MatrixClass[pt].from_npa(_b)
                c = MatrixClass[pt].from_npa(_c)
                d = MatrixClass[pt].from_npa(_d)

                t = time.clock()
                a.add_scaled(context[pt], alpha, b)
                context[pt].synchronize()
                add_scaled_time[pt].append(time.clock() - t)

                t = time.clock()
                a.add(context[pt], b, c, d)
                context[pt].synchronize()
                add_time[pt].append(time.clock() - t)

        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(add_scaled_time['cpu']), 'cpu add_scaled')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(add_scaled_time['gpu']), 'gpu add_scaled')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(add_time['cpu']), 'cpu add')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(add_time['gpu']), 'gpu add')
        print 50 * '-'


def test_add_hprod():
    for shape, N in shapes.iteritems():
        add_hprod_time = {'cpu': [], 'gpu': []}

        for i in xrange(N):
            _a = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _b = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _c = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            alpha = 2 * rng.rand(1)[0] - 1

            for pt in ['cpu', 'gpu']:
                a = MatrixClass[pt].from_npa(_a)
                b = MatrixClass[pt].from_npa(_b)
                c = MatrixClass[pt].from_npa(_c)

                t = time.clock()
                a.add_hprod(context[pt], b, c, alpha)
                context[pt].synchronize()
                add_hprod_time[pt].append(time.clock() - t)

        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(add_hprod_time['cpu']), 'cpu add_hprod')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(add_hprod_time['gpu']), 'gpu add_hprod')
        print 50 * '-'


def test_hprod():
    for shape, N in shapes.iteritems():
        hprod_2_time = {'cpu': [], 'gpu': []}
        hprod_3_time = {'cpu': [], 'gpu': []}

        for i in xrange(N):
            _a = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _b = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _c = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)

            for pt in ['cpu', 'gpu']:
                a = MatrixClass[pt].from_npa(_a)
                b = MatrixClass[pt].from_npa(_b)
                c = MatrixClass[pt].from_npa(_c)
                d = MatrixClass[pt].empty_like(a)

                t = time.clock()
                MatrixClass[pt].assign_hprod(context[pt], d, a, b)
                context[pt].synchronize()
                hprod_2_time[pt].append(time.clock() - t)

                t = time.clock()
                MatrixClass[pt].assign_hprod(context[pt], d, a, b, c)
                context[pt].synchronize()
                hprod_3_time[pt].append(time.clock() - t)

        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(hprod_2_time['cpu']), 'cpu hprod_2')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(hprod_2_time['gpu']), 'gpu hprod_2')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(hprod_3_time['cpu']), 'cpu hprod_3')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(hprod_3_time['gpu']), 'gpu hprod_3')
        print 50 * '-'


def test_sum_hprod():
    for shape, N in shapes.iteritems():
        sum_hprod_4_time = {'cpu': [], 'gpu': []}
        sum_hprod_5_time = {'cpu': [], 'gpu': []}
        sum_hprod_11_time = {'cpu': [], 'gpu': []}

        for _ in xrange(N):
            _a = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _b = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _c = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _d = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _e = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _f = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _g = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _h = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _i = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _j = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _k = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)

            for pt in ['cpu', 'gpu']:
                a = MatrixClass[pt].from_npa(_a)
                b = MatrixClass[pt].from_npa(_b)
                c = MatrixClass[pt].from_npa(_c)
                d = MatrixClass[pt].from_npa(_d)
                e = MatrixClass[pt].from_npa(_e)
                f = MatrixClass[pt].from_npa(_f)
                g = MatrixClass[pt].from_npa(_g)
                h = MatrixClass[pt].from_npa(_h)
                i = MatrixClass[pt].from_npa(_i)
                j = MatrixClass[pt].from_npa(_j)
                k = MatrixClass[pt].from_npa(_k)
                out = MatrixClass[pt].empty_like(a)

                t = time.clock()
                MatrixClass[pt].sum_hprod(context[pt], out, a, b, c, d)
                context[pt].synchronize()
                sum_hprod_4_time[pt].append(time.clock() - t)

                t = time.clock()
                MatrixClass[pt].sum_hprod(context[pt], out, a, b, c, d, e)
                context[pt].synchronize()
                sum_hprod_5_time[pt].append(time.clock() - t)

                t = time.clock()
                MatrixClass[pt].sum_hprod(context[pt], out, a, b, c, d, e, f, g, h, i, j, k)
                context[pt].synchronize()
                sum_hprod_11_time[pt].append(time.clock() - t)

        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(sum_hprod_4_time['cpu']), 'cpu sum_hprod_4')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(sum_hprod_4_time['gpu']), 'gpu sum_hprod_4')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(sum_hprod_5_time['cpu']), 'cpu sum_hprod_5')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(sum_hprod_5_time['gpu']), 'gpu sum_hprod_5')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(sum_hprod_11_time['cpu']), 'cpu sum_hprod_11')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(sum_hprod_11_time['gpu']), 'gpu sum_hprod_11')
        print 50 * '-'


def test_add_dot():
    shape_N = shapes.items()
    vector_shape_N, matrix_shape_N = shape_N[:len(shape_N)/2], shape_N[len(shape_N)/2:]

    for (vector_shape, vector_N), (matrix_shape, matrix_N) in izip(vector_shape_N, matrix_shape_N):
        add_hdot_vector_time = {'N': {'cpu': [], 'gpu': []}, 'T': {'cpu': [], 'gpu': []}}
        add_hdot_matrix_time = {'N': {'cpu': [], 'gpu': []}, 'T': {'cpu': [], 'gpu': []}}

        for i in xrange(matrix_N):
            _a = (4 * rng.rand(vector_shape[0], vector_shape[1]) - 2).astype(dtype=np.float32)
            _b = (4 * rng.rand(matrix_shape[0], matrix_shape[1]) - 2).astype(dtype=np.float32)
            _c = (4 * rng.rand(vector_shape[0], vector_shape[1]) - 2).astype(dtype=np.float32)
            alpha, beta = 2 * rng.rand(2) - 1

            for pt in ['cpu', 'gpu']:
                a = MatrixClass[pt].from_npa(_a)
                b = MatrixClass[pt].from_npa(_b)
                c = MatrixClass[pt].from_npa(_c)

                for matrix_operation in ['N', 'T']:
                    t = time.clock()
                    a.add_dot(context[pt], b, c, matrix_operation, alpha, beta)
                    context[pt].synchronize()
                    add_hdot_vector_time[matrix_operation][pt].append(time.clock() - t)

        for i in xrange(matrix_N):
            _a = (4 * rng.rand(matrix_shape[0], matrix_shape[1]) - 2).astype(dtype=np.float32)
            _b = (4 * rng.rand(matrix_shape[0], matrix_shape[1]) - 2).astype(dtype=np.float32)
            _c = (4 * rng.rand(matrix_shape[0], matrix_shape[1]) - 2).astype(dtype=np.float32)
            alpha, beta = 2 * rng.rand(2) - 1

            for pt in ['cpu', 'gpu']:
                a = MatrixClass[pt].from_npa(_a)
                b = MatrixClass[pt].from_npa(_b)
                c = MatrixClass[pt].from_npa(_c)

                for matrix_operation in ['N', 'T']:
                    t = time.clock()
                    a.add_dot(context[pt], b, c, matrix_operation, alpha, beta)
                    context[pt].synchronize()
                    add_hdot_matrix_time[matrix_operation][pt].append(time.clock() - t)

        print 'matrix_shape: {:4}:{:4} vector_shape: {:4}:{:4} {:.10f} {:20s}'.format(matrix_shape[0], matrix_shape[1], vector_shape[0], vector_shape[1], np.mean(add_hdot_vector_time['N']['cpu']), 'N cpu add_dot vector')
        print 'matrix_shape: {:4}:{:4} vector_shape: {:4}:{:4} {:.10f} {:20s}'.format(matrix_shape[0], matrix_shape[1], vector_shape[0], vector_shape[1], np.mean(add_hdot_vector_time['N']['gpu']), 'N gpu add_dot vector')
        print 'matrix_shape: {:4}:{:4} vector_shape: {:4}:{:4} {:.10f} {:20s}'.format(matrix_shape[0], matrix_shape[1], vector_shape[0], vector_shape[1], np.mean(add_hdot_vector_time['T']['cpu']), 'T cpu add_dot vector')
        print 'matrix_shape: {:4}:{:4} vector_shape: {:4}:{:4} {:.10f} {:20s}'.format(matrix_shape[0], matrix_shape[1], vector_shape[0], vector_shape[1], np.mean(add_hdot_vector_time['T']['gpu']), 'T gpu add_dot vector')
        print 'matrix_shape: {:4}:{:4} {:.10f} {:20s}'.format(matrix_shape[0], matrix_shape[1], np.mean(add_hdot_matrix_time['N']['cpu']), 'N cpu add_dot matrix')
        print 'matrix_shape: {:4}:{:4} {:.10f} {:20s}'.format(matrix_shape[0], matrix_shape[1], np.mean(add_hdot_matrix_time['N']['gpu']), 'N gpu add_dot matrix')
        print 'matrix_shape: {:4}:{:4} {:.10f} {:20s}'.format(matrix_shape[0], matrix_shape[1], np.mean(add_hdot_matrix_time['T']['cpu']), 'T cpu add_dot matrix')
        print 'matrix_shape: {:4}:{:4} {:.10f} {:20s}'.format(matrix_shape[0], matrix_shape[1], np.mean(add_hdot_matrix_time['T']['gpu']), 'T gpu add_dot matrix')
        print 50 * '-'


def test_vdot():
    for shape, N in shapes.iteritems():
        vdot_time = {'cpu': [], 'gpu': []}

        for i in xrange(N):
            _a = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)
            _b = (4 * rng.rand(shape[0], shape[1]) - 2).astype(dtype=np.float32)

            for pt in ['cpu', 'gpu']:
                a = MatrixClass[pt].from_npa(_a)
                b = MatrixClass[pt].from_npa(_b)

                t = time.clock()
                vdot = a.vdot(context[pt], b)
                vdot_time[pt].append(time.clock() - t)

        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(vdot_time['cpu']), 'cpu vdot')
        print 'shape: {:4}:{:4} {:.10f} {:20s}'.format(shape[0], shape[1], np.mean(vdot_time['gpu']), 'gpu vdot')
        print 50 * '-'


def test_lstm_block():
    z_context = {'cpu': MatrixContextClass['cpu'](), 'gpu': MatrixContextClass['gpu']()}
    i_context = {'cpu': MatrixContextClass['cpu'](), 'gpu': MatrixContextClass['gpu']()}
    f_context = {'cpu': MatrixContextClass['cpu'](), 'gpu': MatrixContextClass['gpu']()}
    c_context = {'cpu': MatrixContextClass['cpu'](), 'gpu': MatrixContextClass['gpu']()}
    o_context = {'cpu': MatrixContextClass['cpu'](), 'gpu': MatrixContextClass['gpu']()}
    shape_N = shapes.items()
    matrix_shape_N = shape_N[len(shape_N)/2:]

    for shape, N in matrix_shape_N:
        lstm_block_forward_time = {'cpu': [], 'gpu': []}

        for i in xrange(N):
            _Wz = 0.5 * (2 * rng.rand(shape[0], shape[1]) - 1).astype(dtype=np.float32)
            _Rz = 0.5 * (2 * rng.rand(shape[0], shape[1]) - 1).astype(dtype=np.float32)
            _Wi = 0.5 * (2 * rng.rand(shape[0], shape[1]) - 1).astype(dtype=np.float32)
            _Ri = 0.5 * (2 * rng.rand(shape[0], shape[1]) - 1).astype(dtype=np.float32)
            _pi = 0.5 * (2 * rng.rand(shape[0], 1) - 1).astype(dtype=np.float32)
            _Wf = 0.5 * (2 * rng.rand(shape[0], shape[1]) - 1).astype(dtype=np.float32)
            _Rf = 0.5 * (2 * rng.rand(shape[0], shape[1]) - 1).astype(dtype=np.float32)
            _pf = 0.5 * (2 * rng.rand(shape[0], 1) - 1).astype(dtype=np.float32)
            _Wo = 0.5 * (2 * rng.rand(shape[0], shape[1]) - 1).astype(dtype=np.float32)
            _Ro = 0.5 * (2 * rng.rand(shape[0], shape[1]) - 1).astype(dtype=np.float32)
            _po = 0.5 * (2 * rng.rand(shape[0], 1) - 1).astype(dtype=np.float32)
            _pre_z_t = 0.5 * (2 * rng.rand(shape[0], 1) - 1).astype(dtype=np.float32)
            _pre_i_t = 0.5 * (2 * rng.rand(shape[0], 1) - 1).astype(dtype=np.float32)
            _pre_f_t = 0.5 * (2 * rng.rand(shape[0], 1) - 1).astype(dtype=np.float32)
            _pre_o_t = 0.5 * (2 * rng.rand(shape[0], 1) - 1).astype(dtype=np.float32)
            _h_tm1 = 0.5 * (2 * rng.rand(shape[0], 1) - 1).astype(dtype=np.float32)
            _c_tm1 = 0.5 * (2 * rng.rand(shape[0], 1) - 1).astype(dtype=np.float32)

            for pt in ['cpu', 'gpu']:
                Wz = MatrixClass[pt].from_npa(_Wz)
                Rz = MatrixClass[pt].from_npa(_Rz)
                Wi = MatrixClass[pt].from_npa(_Wi)
                Ri = MatrixClass[pt].from_npa(_Ri)
                pi = MatrixClass[pt].from_npa(_pi)
                Wf = MatrixClass[pt].from_npa(_Wf)
                Rf = MatrixClass[pt].from_npa(_Rf)
                pf = MatrixClass[pt].from_npa(_pf)
                Wo = MatrixClass[pt].from_npa(_Wo)
                Ro = MatrixClass[pt].from_npa(_Ro)
                po = MatrixClass[pt].from_npa(_po)

                c_t = MatrixClass[pt].empty_like(po)
                h_t = MatrixClass[pt].empty_like(po)

                LstmBlock.p_type = pt
                lstm_block = LstmBlock(Wz, Rz, Wi, Ri, pi, Wf, Rf, pf, Wo, Ro, po, c_t, h_t,
                                       None, None, None, None,
                                       z_context[pt], i_context[pt], f_context[pt], c_context[pt], o_context[pt])

                pre_z_t = MatrixClass[pt].from_npa(_pre_z_t)
                pre_i_t = MatrixClass[pt].from_npa(_pre_i_t)
                pre_f_t = MatrixClass[pt].from_npa(_pre_f_t)
                pre_o_t = MatrixClass[pt].from_npa(_pre_o_t)
                h_tm1 = MatrixClass[pt].from_npa(_h_tm1)
                c_tm1 = MatrixClass[pt].from_npa(_c_tm1)

                lstm_block.set_training_mode()
                t = time.clock()
                lstm_block.fprop(pre_z_t, pre_i_t, pre_f_t, pre_o_t, h_tm1, c_tm1)
                lstm_block_forward_time[pt].append(time.clock() - t)

        print 'shape: {:4}: {:.10f} {:20s}'.format(shape[0], np.mean(lstm_block_forward_time['cpu']), 'cpu lstm_block_forward')
        print 'shape: {:4}: {:.10f} {:20s}'.format(shape[0], np.mean(lstm_block_forward_time['gpu']), 'gpu lstm_block_forward')
        print 50 * '-'


if __name__ == '__main__':
    # print '{}{}{}'.format(40 * '=', 'test_scale', 40 * '=')
    # test_scale()
    # print '{}{}{}'.format(40 * '=', 'test_tanh', 40 * '=')
    # test_tanh()
    # print '{}{}{}'.format(40 * '=', 'test_sigmoid', 40 * '=')
    # test_sigmoid()
    # print '{}{}{}'.format(40 * '=', 'test_add', 40 * '=')
    # test_add()
    # print '{}{}{}'.format(40 * '=', 'test_add_hprod', 40 * '=')
    # test_add_hprod()
    # print '{}{}{}'.format(40 * '=', 'test_hprod', 40 * '=')
    # test_hprod()
    # print '{}{}{}'.format(40 * '=', 'test_sum_hprod', 40 * '=')
    # test_sum_hprod()
    # print '{}{}{}'.format(40 * '=', 'test_add_dot', 40 * '=')
    # test_add_dot()
    # print '{}{}{}'.format(40 * '=', 'test_vdot', 40 * '=')
    # test_vdot()
    print '{}{}{}'.format(40 * '=', 'test_lstm_block', 40 * '=')
    test_lstm_block()