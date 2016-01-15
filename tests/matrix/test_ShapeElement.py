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
import numpy as np
from unittest import TestCase
from quagga.context import GpuContext
from quagga.context import CpuContext
from quagga.matrix import ShapeElement


class TestMatrix(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(seed=42)
        cls.gpu_context = GpuContext()
        cls.cpu_context = CpuContext()
        cls.N = 1000
        cls.max_int = 1000

    def test_add(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = self.rng.randint(self.max_int)
            b_se = ShapeElement(b)
            delta = self.rng.randint(self.max_int)

            c_se = a_se + b
            r.append(c_se.value == a + b)
            a_se[:] = a + delta
            r.append(c_se.value == a + b + delta)
            a_se[:] = a
            c_se = a_se + b_se
            r.append(c_se.value == a + b)
            a_se[:] = a + delta
            b_se[:] = b + delta
            r.append(c_se.value == a + b + 2 * delta)
        self.assertTrue(all(r))

    def test_sub(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = self.rng.randint(self.max_int)
            b_se = ShapeElement(b)
            delta = self.rng.randint(self.max_int)

            c_se = a_se - b
            r.append(c_se.value == a - b)
            a_se[:] = a + delta
            r.append(c_se.value == a - b + delta)
            a_se[:] = a
            c_se = a_se - b_se
            r.append(c_se.value == a - b)
            a_se[:] = a + delta
            b_se[:] = b - delta
            r.append(c_se.value == a - b + 2 * delta)
        self.assertTrue(all(r))

    def test_mul(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = self.rng.randint(self.max_int)
            b_se = ShapeElement(b)
            delta = self.rng.randint(self.max_int)

            c_se = a_se * b
            r.append(c_se.value == a * b)
            a_se[:] = a * delta
            r.append(c_se.value == a * b * delta)
            a_se[:] = a
            c_se = a_se * b_se
            r.append(c_se.value == a * b)
            a_se[:] = a * delta
            b_se[:] = b * delta
            r.append(c_se.value == a * b * delta * delta)
        self.assertTrue(all(r))

    def test_div(self):
        r = []
        for _ in xrange(self.N):
            a = 1 + self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = 1 + self.rng.randint(self.max_int)
            b_se = ShapeElement(b)
            delta = 1 + self.rng.randint(self.max_int)

            c_se = a_se / b
            r.append(c_se.value == a / b)
            a_se[:] = a * delta
            r.append(c_se.value == a * delta / b)
            a_se[:] = a
            c_se = a_se / b_se
            r.append(c_se.value == a / b)
            a_se[:] = a + delta
            b_se[:] = b + delta
            r.append(c_se.value == (a + delta) / (b + delta))
        self.assertTrue(all(r))

    def test_radd(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = self.rng.randint(self.max_int)
            delta = self.rng.randint(self.max_int)

            c_se = b + a_se
            r.append(c_se.value == a + b)
            a_se[:] = a + delta
            r.append(c_se.value == a + b + delta)
        self.assertTrue(all(r))

    def test_rsub(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = self.rng.randint(self.max_int)
            delta = self.rng.randint(self.max_int)

            c_se = b - a_se
            r.append(c_se.value == b - a)
            a_se[:] = a + delta
            r.append(c_se.value == b - a - delta)
        self.assertTrue(all(r))

    def test_rmul(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = self.rng.randint(self.max_int)
            delta = self.rng.randint(self.max_int)

            c_se = b * a_se
            r.append(c_se.value == a * b)
            a_se[:] = a * delta
            r.append(c_se.value == a * b * delta)
        self.assertTrue(all(r))

    def test_rdiv(self):
        r = []
        for _ in xrange(self.N):
            a = 1 + self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = self.rng.randint(self.max_int)
            delta = 1 + self.rng.randint(self.max_int)

            c_se = b / a_se
            r.append(c_se.value == b / a)
            try:
                a_se[:] = a - delta + (a - delta == 0)
                r.append(c_se.value == b / (a - delta + (a - delta == 0)))
            except:
                pass
        self.assertTrue(all(r))

    def test_eq(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = self.rng.randint(self.max_int)
            b_se = ShapeElement(b)

            r.append((a == a_se) == (a == a))
            r.append((a_se == a) == (a == a))
            r.append((a_se == a_se) == (a == a))
            r.append((a_se == b) == (a == b))
            r.append((a_se == b_se) == (a == b))
            r.append((a == b_se) == (a == b))
        self.assertTrue(all(r))

    def test_ne(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = self.rng.randint(self.max_int)
            b_se = ShapeElement(b)

            r.append((a != a_se) == (a != a))
            r.append((a_se != a) == (a != a))
            r.append((a_se != a_se) == (a != a))
            r.append((a_se != b) == (a != b))
            r.append((a_se != b_se) == (a != b))
            r.append((a != b_se) == (a != b))
        self.assertTrue(all(r))

    def test_lt(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = self.rng.randint(self.max_int)
            b_se = ShapeElement(b)

            r.append((a < a_se) == (a < a))
            r.append((a_se < a) == (a < a))
            r.append((a_se < a_se) == (a < a))
            r.append((a_se < b) == (a < b))
            r.append((a_se < b_se) == (a < b))
            r.append((a < b_se) == (a < b))
        self.assertTrue(all(r))

    def test_gt(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = self.rng.randint(self.max_int)
            b_se = ShapeElement(b)

            r.append((a > a_se) == (a > a))
            r.append((a_se > a) == (a > a))
            r.append((a_se > a_se) == (a > a))
            r.append((a_se > b) == (a > b))
            r.append((a_se > b_se) == (a > b))
            r.append((a > b_se) == (a > b))
        self.assertTrue(all(r))

    def test_le(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = self.rng.randint(self.max_int)
            b_se = ShapeElement(b)

            r.append((a <= a_se) == (a <= a))
            r.append((a_se <= a) == (a <= a))
            r.append((a_se <= a_se) == (a <= a))
            r.append((a_se <= b) == (a <= b))
            r.append((a_se <= b_se) == (a <= b))
            r.append((a <= b_se) == (a <= b))
        self.assertTrue(all(r))

    def test_ge(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            b = self.rng.randint(self.max_int)
            b_se = ShapeElement(b)

            r.append((a >= a_se) == (a >= a))
            r.append((a_se >= a) == (a >= a))
            r.append((a_se >= a_se) == (a >= a))
            r.append((a_se >= b) == (a >= b))
            r.append((a_se >= b_se) == (a >= b))
            r.append((a >= b_se) == (a >= b))
        self.assertTrue(all(r))

    def test_str(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            r.append(str(a_se) == str(a))
        self.assertTrue(all(r))

    def test_int(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            r.append(int(a_se) == a)
        self.assertTrue(all(r))

    def test_float(self):
        r = []
        for _ in xrange(self.N):
            a = self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            r.append(float(a_se) == float(a))
        self.assertTrue(all(r))

    def test_index(self):
        r = []
        for _ in xrange(self.N):
            a = 1 + self.rng.randint(self.max_int)
            a_se = ShapeElement(a)
            l = [42] * 2 * a
            r.append(l[a_se] == l[a])
            with self.assertRaises(ValueError):
                a_se = ShapeElement(float(a))
                r.append(l[a_se] == l[a])
        self.assertTrue(all(r))