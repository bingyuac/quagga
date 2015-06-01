import ctypes
import numpy as np
from unittest import TestCase
from quagga.cuda import cudart
from quagga.matrix import GpuMatrix, GpuMatrixContext, CpuMatrix, CpuMatrixContext


class TestMatrix(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.gpu_context = GpuMatrixContext()
        cls.cpu_context = CpuMatrixContext()
        cls.N = 50

    @classmethod
    def get_random_array(cls, shape=None):
        if shape:
            a = 4 * cls.rng.rand(*shape) - 2
        else:
            nrows, ncols = cls.rng.randint(low=1, high=7000, size=2)
            a = 4 * cls.rng.rand(nrows, ncols) - 2
        return a.astype(dtype=np.float32)

    def test_from_numpy_array(self):
        r = []
        for i in xrange(self.N):
            a = TestMatrix.get_random_array()
            a_gpu = GpuMatrix.from_npa(a)
            r.append(np.allclose(a, a_gpu.to_host()))
        self.assertEqual(sum(r), self.N)

    def test_getitem(self):
        r = []
        for i in xrange(self.N):
            a = TestMatrix.get_random_array()
            j = self.rng.randint(a.shape[1])
            option = self.rng.randint(3)
            if option == 0:
                s = j
            elif option == 1:
                s = slice(None, j, None)
            else:
                s = slice(j, None, None)

            a_cpu = CpuMatrix.from_npa(a)
            a_gpu = GpuMatrix.from_npa(a)
            a_cpu_column = a_cpu[:, s]
            a_gpu_column = a_gpu[:, s]
            r.append(np.allclose(a_cpu_column.to_host(),
                                 a_gpu_column.to_host()))
        self.assertEqual(sum(r), self.N)

    def test_scale(self):
        r = []
        for i in xrange(self.N):
            a = TestMatrix.get_random_array()
            alpha = 2 * self.rng.rand(1)[0] - 1

            a_cpu = CpuMatrix.from_npa(a)
            out_cpu = CpuMatrix.empty_like(a_cpu)
            a_gpu = GpuMatrix.from_npa(a)
            out_gpu = GpuMatrix.empty_like(a_gpu)

            a_cpu.scale(self.cpu_context, alpha, out_cpu)
            a_gpu.scale(self.gpu_context, alpha, out_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(out_cpu.to_host(), out_gpu.to_host()))

            a_cpu.scale(self.cpu_context, alpha)
            a_gpu.scale(self.gpu_context, alpha)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(a_cpu.to_host(), out_gpu.to_host()))

        self.assertEqual(sum(r), self.N * 2)

    def test_tanh(self):
        r = []
        for i in xrange(self.N):
            a = TestMatrix.get_random_array()

            a_cpu = CpuMatrix.from_npa(a)
            tanh_matrix_cpu = CpuMatrix.empty_like(a_cpu)
            derivative_matrix_cpu = CpuMatrix.empty_like(a_cpu)
            a_gpu = GpuMatrix.from_npa(a)
            tanh_matrix_gpu = GpuMatrix.empty_like(a_gpu)
            derivative_matrix_gpu = GpuMatrix.empty_like(a_gpu)

            a_cpu.tanh(self.cpu_context, tanh_matrix_cpu)
            a_gpu.tanh(self.gpu_context, tanh_matrix_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(tanh_matrix_cpu.to_host(),
                                 tanh_matrix_gpu.to_host()))

            a_cpu.tanh(self.cpu_context, tanh_matrix_cpu, derivative_matrix_cpu)
            a_gpu.tanh(self.gpu_context, tanh_matrix_gpu, derivative_matrix_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(tanh_matrix_cpu.to_host(),
                                 tanh_matrix_gpu.to_host()))
            r.append(np.allclose(derivative_matrix_cpu.to_host(),
                                 derivative_matrix_gpu.to_host()))

        self.assertEqual(sum(r), self.N * 3)

    def test_sigmoid(self):
        r = []
        for i in xrange(self.N):
            a = TestMatrix.get_random_array()

            a_cpu = CpuMatrix.from_npa(a)
            sigmoid_matrix_cpu = CpuMatrix.empty_like(a_cpu)
            derivative_matrix_cpu = CpuMatrix.empty_like(a_cpu)
            a_gpu = GpuMatrix.from_npa(a)
            sigmoid_matrix_gpu = GpuMatrix.empty_like(a_gpu)
            derivative_matrix_gpu = GpuMatrix.empty_like(a_gpu)

            a_cpu.sigmoid(self.cpu_context, sigmoid_matrix_cpu)
            a_gpu.sigmoid(self.gpu_context, sigmoid_matrix_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(sigmoid_matrix_cpu.to_host(),
                                 sigmoid_matrix_gpu.to_host()))

            a_cpu.sigmoid(self.cpu_context, sigmoid_matrix_cpu, derivative_matrix_cpu)
            a_gpu.sigmoid(self.gpu_context, sigmoid_matrix_gpu, derivative_matrix_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(sigmoid_matrix_cpu.to_host(),
                                 sigmoid_matrix_gpu.to_host()))
            r.append(np.allclose(derivative_matrix_cpu.to_host(),
                                 derivative_matrix_gpu.to_host()))

        self.assertEqual(sum(r), self.N * 3)

    def test_add_scaled(self):
        r = []
        for i in xrange(self.N):
            a = TestMatrix.get_random_array()
            b = TestMatrix.get_random_array(a.shape)
            alpha = 2 * self.rng.rand(1)[0] - 1

            a_cpu = CpuMatrix.from_npa(a)
            b_cpu = CpuMatrix.from_npa(b)
            a_gpu = GpuMatrix.from_npa(a)
            b_gpu = GpuMatrix.from_npa(b)

            a_cpu.add(self.cpu_context, b_cpu)
            a_gpu.add(self.gpu_context, b_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(a_cpu.to_host(), a_gpu.to_host(), atol=1e-6))

            a_cpu.add_scaled(self.cpu_context, alpha, b_cpu)
            a_gpu.add_scaled(self.gpu_context, alpha, b_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(a_cpu.to_host(), a_gpu.to_host(), atol=1e-6))

        self.assertEqual(sum(r), self.N * 2)

    def test_add(self):
        r = []
        for i in xrange(self.N):
            a = TestMatrix.get_random_array()
            b = TestMatrix.get_random_array(a.shape)
            c = TestMatrix.get_random_array(a.shape)
            d = TestMatrix.get_random_array(a.shape)

            a_cpu = CpuMatrix.from_npa(a)
            b_cpu = CpuMatrix.from_npa(b)
            c_cpu = CpuMatrix.from_npa(c)
            d_cpu = CpuMatrix.from_npa(d)
            a_gpu = GpuMatrix.from_npa(a)
            b_gpu = GpuMatrix.from_npa(b)
            c_gpu = GpuMatrix.from_npa(c)
            d_gpu = GpuMatrix.from_npa(d)

            a_cpu.add(self.cpu_context, b_cpu, c_cpu, d_cpu)
            a_gpu.add(self.gpu_context, b_gpu, c_gpu, d_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(a_cpu.to_host(), a_gpu.to_host(), atol=1e-6))

        self.assertEqual(sum(r), self.N)

    def test_sliced_add(self):
        r = []
        for i in xrange(self.N):
            a = TestMatrix.get_random_array()
            b = TestMatrix.get_random_array((a.shape[0], a.shape[1]+10000))
            alpha = 2 * self.rng.rand(1)[0] - 1

            a_cpu = CpuMatrix.from_npa(a)
            b_cpu = CpuMatrix.from_npa(b)
            column_indxs_cpu = self.rng.choice(b.shape[1], a.shape[1]).astype(dtype=np.int32)
            a_gpu = GpuMatrix.from_npa(a)
            b_gpu = GpuMatrix.from_npa(b)
            raw_column_indxs = column_indxs_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            nbytes = column_indxs_cpu.size * ctypes.sizeof(ctypes.c_int)
            column_indxs_gpu = cudart.cuda_malloc(nbytes, ctypes.c_int)
            cudart.cuda_memcpy(column_indxs_gpu, raw_column_indxs, nbytes, 'host_to_device')

            b_cpu.sliced_add(self.cpu_context, a_cpu, column_indxs_cpu, alpha)
            b_gpu.sliced_add(self.gpu_context, a_gpu, column_indxs_gpu, alpha)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(b_cpu.to_host(), b_gpu.to_host(), atol=1e-6))

        self.assertEqual(sum(r), self.N)

    def test_add_hprod(self):
        r = []
        for i in xrange(self.N):
            a = TestMatrix.get_random_array()
            b = TestMatrix.get_random_array(a.shape)
            c = TestMatrix.get_random_array(a.shape)
            alpha = 2 * self.rng.rand(1)[0] - 1

            a_cpu = CpuMatrix.from_npa(a)
            b_cpu = CpuMatrix.from_npa(b)
            c_cpu = CpuMatrix.from_npa(c)
            a_gpu = GpuMatrix.from_npa(a)
            b_gpu = GpuMatrix.from_npa(b)
            c_gpu = GpuMatrix.from_npa(c)

            a_cpu.add_hprod(self.cpu_context, b_cpu, c_cpu, alpha)
            a_gpu.add_hprod(self.gpu_context, b_gpu, c_gpu, alpha)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(a_cpu.to_host(), a_gpu.to_host(), atol=1e-6))

        self.assertEqual(sum(r), self.N)

    def test_assign_hprod(self):
        r = []
        for i in xrange(self.N):
            a = TestMatrix.get_random_array()
            b = TestMatrix.get_random_array(a.shape)
            c = TestMatrix.get_random_array(a.shape)

            a_cpu = CpuMatrix.from_npa(a)
            b_cpu = CpuMatrix.from_npa(b)
            c_cpu = CpuMatrix.from_npa(c)
            out_cpu = CpuMatrix.empty_like(c_cpu)
            a_gpu = GpuMatrix.from_npa(a)
            b_gpu = GpuMatrix.from_npa(b)
            c_gpu = GpuMatrix.from_npa(c)
            out_gpu = GpuMatrix.empty_like(c_gpu)

            out_cpu.assign_hprod(self.cpu_context, a_cpu, b_cpu, c_cpu)
            out_gpu.assign_hprod(self.gpu_context, a_gpu, b_gpu, c_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(out_cpu.to_host(), out_gpu.to_host()))

            out_cpu.assign_hprod(self.cpu_context, a_cpu, b_cpu)
            out_gpu.assign_hprod(self.gpu_context, a_gpu, b_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(out_cpu.to_host(), out_gpu.to_host()))

        self.assertEqual(sum(r), self.N * 2)

    def test_assign_sum_hprod(self):
        r = []
        for _ in xrange(self.N):
            a = TestMatrix.get_random_array()
            b = TestMatrix.get_random_array(a.shape)
            c = TestMatrix.get_random_array(a.shape)
            d = TestMatrix.get_random_array(a.shape)
            e = TestMatrix.get_random_array(a.shape)
            f = TestMatrix.get_random_array(a.shape)
            g = TestMatrix.get_random_array(a.shape)
            h = TestMatrix.get_random_array(a.shape)
            i = TestMatrix.get_random_array(a.shape)
            j = TestMatrix.get_random_array(a.shape)
            k = TestMatrix.get_random_array(a.shape)

            a_cpu = CpuMatrix.from_npa(a)
            b_cpu = CpuMatrix.from_npa(b)
            c_cpu = CpuMatrix.from_npa(c)
            d_cpu = CpuMatrix.from_npa(d)
            e_cpu = CpuMatrix.from_npa(e)
            f_cpu = CpuMatrix.from_npa(f)
            g_cpu = CpuMatrix.from_npa(g)
            h_cpu = CpuMatrix.from_npa(h)
            i_cpu = CpuMatrix.from_npa(i)
            j_cpu = CpuMatrix.from_npa(j)
            k_cpu = CpuMatrix.from_npa(k)
            out_cpu = CpuMatrix.empty_like(c_cpu)
            a_gpu = GpuMatrix.from_npa(a)
            b_gpu = GpuMatrix.from_npa(b)
            c_gpu = GpuMatrix.from_npa(c)
            d_gpu = GpuMatrix.from_npa(d)
            e_gpu = GpuMatrix.from_npa(e)
            f_gpu = GpuMatrix.from_npa(f)
            g_gpu = GpuMatrix.from_npa(g)
            h_gpu = GpuMatrix.from_npa(h)
            i_gpu = GpuMatrix.from_npa(i)
            j_gpu = GpuMatrix.from_npa(j)
            k_gpu = GpuMatrix.from_npa(k)
            out_gpu = GpuMatrix.empty_like(c_gpu)

            out_cpu.assign_sum_hprod(self.cpu_context, a_cpu, b_cpu, c_cpu, d_cpu)
            out_gpu.assign_sum_hprod(self.gpu_context, a_gpu, b_gpu, c_gpu, d_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(out_cpu.to_host(), out_gpu.to_host(), atol=1e-6))

            out_cpu.assign_sum_hprod(self.cpu_context, a_cpu, b_cpu, c_cpu, d_cpu, e_cpu)
            out_gpu.assign_sum_hprod(self.gpu_context, a_gpu, b_gpu, c_gpu, d_gpu, e_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(out_cpu.to_host(), out_gpu.to_host(), atol=1e-6))

            out_cpu.assign_sum_hprod(self.cpu_context, a_cpu, b_cpu, c_cpu, d_cpu, e_cpu, f_cpu, g_cpu, h_cpu, i_cpu, j_cpu, k_cpu)
            out_gpu.assign_sum_hprod(self.gpu_context, a_gpu, b_gpu, c_gpu, d_gpu, e_gpu, f_gpu, g_gpu, h_gpu, i_gpu, j_gpu, k_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(out_cpu.to_host(), out_gpu.to_host(), atol=1e-5))

        self.assertEqual(sum(r), self.N * 3)

    def test_assign_hprod_sum(self):
        r = []
        for _ in xrange(self.N):
            a = TestMatrix.get_random_array()
            b = TestMatrix.get_random_array(a.shape)

            a_cpu = CpuMatrix.from_npa(a)
            b_cpu = CpuMatrix.from_npa(b)
            out_cpu = CpuMatrix.empty(a_cpu.nrows, 1)

            a_gpu = GpuMatrix.from_npa(a)
            b_gpu = GpuMatrix.from_npa(b)
            out_gpu = GpuMatrix.empty(a_cpu.nrows, 1)

            out_cpu.assign_hprod_sum(self.cpu_context, a_cpu, b_cpu)
            out_gpu.assign_hprod_sum(self.gpu_context, a_gpu, b_gpu)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(out_cpu.to_host(), out_gpu.to_host(), atol=1e-3))

        self.assertEqual(sum(r), self.N)

    def test_assign_dot(self):
        r = []
        for i in xrange(self.N):
            a = TestMatrix.get_random_array()
            a_v = TestMatrix.get_random_array((a.shape[0], 1))
            m = self.rng.randint(low=1, high=2000, size=1)[0]
            mat_op_b = self.rng.choice(['T', 'N'], 1)[0]
            mat_op_c = self.rng.choice(['T', 'N'], 1)[0]
            if mat_op_b == 'N':
                b = TestMatrix.get_random_array((a.shape[0], m))
            else:
                b = TestMatrix.get_random_array((m, a.shape[0]))
            if mat_op_c == 'N':
                c = TestMatrix.get_random_array((m, a.shape[1]))
                c_v = TestMatrix.get_random_array((m, 1))
            else:
                c = TestMatrix.get_random_array((a.shape[1], m))
                c_v = TestMatrix.get_random_array((1, m))

            a_cpu = CpuMatrix.from_npa(a)
            a_v_cpu = CpuMatrix.from_npa(a_v)
            b_cpu = CpuMatrix.from_npa(b)
            c_cpu = CpuMatrix.from_npa(c)
            c_v_cpu = CpuMatrix.from_npa(c_v)
            a_gpu = GpuMatrix.from_npa(a)
            a_v_gpu = GpuMatrix.from_npa(a_v)
            b_gpu = GpuMatrix.from_npa(b)
            c_gpu = GpuMatrix.from_npa(c)
            c_v_gpu = GpuMatrix.from_npa(c_v)

            a_cpu.assign_dot(self.cpu_context, b_cpu, c_cpu, mat_op_b, mat_op_c)
            a_gpu.assign_dot(self.gpu_context, b_gpu, c_gpu, mat_op_b, mat_op_c)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(a_cpu.to_host(), a_gpu.to_host(), atol=1e-3))

            a_v_cpu.assign_dot(self.cpu_context, b_cpu, c_v_cpu, mat_op_b, mat_op_c)
            a_v_gpu.assign_dot(self.gpu_context, b_gpu, c_v_gpu, mat_op_b, mat_op_c)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(a_v_cpu.to_host(), a_v_gpu.to_host(), atol=1e-3))

        self.assertEqual(sum(r), self.N * 2)

    def test_add_dot(self):
        r = []
        for i in xrange(self.N):
            a = TestMatrix.get_random_array()
            a_v = TestMatrix.get_random_array((a.shape[0], 1))
            m = self.rng.randint(low=1, high=2000, size=1)[0]
            mat_op_b = self.rng.choice(['T', 'N'], 1)[0]
            mat_op_c = self.rng.choice(['T', 'N'], 1)[0]
            if mat_op_b == 'N':
                b = TestMatrix.get_random_array((a.shape[0], m))
            else:
                b = TestMatrix.get_random_array((m, a.shape[0]))
            if mat_op_c == 'N':
                c = TestMatrix.get_random_array((m, a.shape[1]))
                c_v = TestMatrix.get_random_array((m, 1))
            else:
                c = TestMatrix.get_random_array((a.shape[1], m))
                c_v = TestMatrix.get_random_array((1, m))
            alpha = 2 * self.rng.rand(1)[0] - 1
            beta = 2 * self.rng.rand(1)[0] - 1

            a_cpu = CpuMatrix.from_npa(a)
            a_v_cpu = CpuMatrix.from_npa(a_v)
            b_cpu = CpuMatrix.from_npa(b)
            c_cpu = CpuMatrix.from_npa(c)
            c_v_cpu = CpuMatrix.from_npa(c_v)
            a_gpu = GpuMatrix.from_npa(a)
            a_v_gpu = GpuMatrix.from_npa(a_v)
            b_gpu = GpuMatrix.from_npa(b)
            c_gpu = GpuMatrix.from_npa(c)
            c_v_gpu = GpuMatrix.from_npa(c_v)

            a_cpu.add_dot(self.cpu_context, b_cpu, c_cpu, mat_op_b, mat_op_c, alpha, beta)
            a_gpu.add_dot(self.gpu_context, b_gpu, c_gpu, mat_op_b, mat_op_c, alpha, beta)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(a_cpu.to_host(), a_gpu.to_host(), atol=1e-3))

            a_v_cpu.add_dot(self.cpu_context, b_cpu, c_v_cpu, mat_op_b, mat_op_c, alpha, beta)
            a_v_gpu.add_dot(self.gpu_context, b_gpu, c_v_gpu, mat_op_b, mat_op_c, alpha, beta)
            self.cpu_context.synchronize()
            self.gpu_context.synchronize()
            r.append(np.allclose(a_v_cpu.to_host(), a_v_gpu.to_host(), atol=1e-3))

        self.assertEqual(sum(r), self.N * 2)

    def test_vdot(self):
        r = []
        for i in xrange(self.N):
            m = self.rng.randint(low=1, high=7000, size=1)[0]
            a = TestMatrix.get_random_array((m, 1))
            b = TestMatrix.get_random_array((m, 1))

            a_cpu = CpuMatrix.from_npa(a)
            b_cpu = CpuMatrix.from_npa(b)
            a_gpu = GpuMatrix.from_npa(a)
            b_gpu = GpuMatrix.from_npa(b)

            r.append(np.allclose(a_cpu.vdot(self.cpu_context, b_cpu),
                                 a_gpu.vdot(self.gpu_context, b_gpu), atol=1e-4))

        self.assertEqual(sum(r), self.N)