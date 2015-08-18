import numpy as np
import ctypes as ct
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class NpLstmRnn(object):
    def __init__(self, W_init, R_init, x, learning=True, device_id=None):
        """
        TODO
        """
        if W_init.nrows != R_init.nrows:
            raise ValueError('W and R must have the same number of rows!')
        if R_init.nrows != R_init.ncols:
            raise ValueError('R must be a square matrix!')

        nrows = R_init.nrows
        self.context = Context(device_id)
        self.max_input_sequence_len = x.ncols

        W = [Matrix.from_npa(W_init(), device_id=device_id) for _ in xrange(4)]
        self.W = Matrix.empty(4 * nrows, W_init.ncols, W[0].dtype, device_id)
        self.W.assign_vstack(self.context, W)
        if learning:
            self.dL_dW = Matrix.empty_like(self.W)

        R = [Matrix.from_npa(R_init(), device_id=device_id) for _ in xrange(4)]
        self.R = Matrix.empty(4 * nrows, R_init.ncols, R[0].dtype, device_id)
        self.R.assign_vstack(self.context, R)
        if learning:
            self.dL_dR = Matrix.empty_like(self.R)

        if learning and x._b_usage_context:
            self.x, self.dL_dx = x.register_usage(self.context, self.context)
            self.propagate_to_input = True
        else:
            self.x = x.register_usage(self.context)
            self.propagate_to_input = False

        h = Matrix.empty(nrows, self.max_input_sequence_len, x.dtype, device_id)
        if learning:
            self.h = Connector(h, self.context, self.context)
        else:
            self.h = Connector(h, self.context)

        self.pre_zifo = Matrix.empty(4 * nrows, self.max_input_sequence_len, x.dtype, device_id)
        if learning:
            self.dL_dpre_zifo = Matrix.empty_like(self.pre_zifo, device_id)

        self.lstm_cells = []
        for k in xrange(self.max_input_sequence_len):
            if k == 0:
                prev_c = Matrix.from_npa(np.zeros((nrows, 1)), x.dtype, device_id)
                prev_h = prev_c
            else:
                prev_c = self.lstm_cells[-1].c
                prev_h = self.lstm_cells[-1].h
            if learning:
                cell = _NpLstmCell(self.R, h[:, k], self.pre_zifo[:, k], self.dL_dpre_zifo[:, k], prev_c, prev_h, self.context, learning)
            else:
                cell = _NpLstmCell(self.R, h[:, k], self.pre_zifo[:, k], None, prev_c, prev_h, self.context, learning)
            self.lstm_cells.append(cell)
        self.learning = learning

    def fprop(self):
        n = self.x.ncols
        if n > self.max_input_sequence_len:
            raise ValueError('Sequence has length: {} that is too long. '
                             'The maximum is: {}'.
                             format(n, self.max_input_sequence_len))
        self.h.ncols = n
        self.pre_zifo.ncols = n
        if self.learning:
            self.dL_dpre_zifo.ncols = n
        if self.propagate_to_input:
            self.dL_dx.ncols = n

        self.pre_zifo.assign_dot(self.context, self.W, self.x)
        for k in xrange(n):
            self.lstm_cells[k].fprop()
        self.h.fprop()

    def bprop(self):
        dL_dh = self.h.backward_matrix
        n = dL_dh.ncols
        self.lstm_cells[n-1].dL_dc.scale(self.context, ct.c_float(0.0))
        for k in reversed(xrange(n)):
            if k == 0:
                dL_dprev_h = None
                dL_dprev_c = None
            else:
                dL_dprev_h = dL_dh[:, k-1]
                dL_dprev_c = self.lstm_cells[k-1].dL_dc
            self.lstm_cells[k].bprop(self.context, dL_dh[:, k], dL_dprev_h, dL_dprev_c)

        # dL/dx = W.T * dL/dpre_zifo
        # dL_dW = dL/dpre_zifo * x.T
        # dL_dR = dL/dpre_zifo[:, 1:n] * h[:, :n-1].T
        if self.propagate_to_input:
            self.dL_dx.assign_dot(self.context, self.W, self.dL_dpre_zifo, 'T')
        self.dL_dW.assign_dot(self.context, self.dL_dpre_zifo, self.x, 'N', 'T')
        self.dL_dR.assign_dot(self.context, self.dL_dpre_zifo[:, 1:n], self.h.__getitem__((slice(None), slice(n-1))), 'N', 'T')

    @property
    def params(self):
        return [self.W, self.R]

    @property
    def grads(self):
        return [self.dL_dW, self.dL_dR]


class _NpLstmCell(object):
    def __init__(self, R, h, pre_zifo, dL_dpre_zifo, prev_c, prev_h, context, learning=True):
        """
        No peepholes LSTM cell block is used for building `NpLstmRnn` block.
        This block is not completely autonomous it requires precomputed
        `W * x` -- pre_zifo which is not the connector. That is why `NpLstmRnn`
        should take care of proper synchronization.

        :param R: matrix that contains vertically stacked Rz, Ri, Rf, Ro
        :param h: preallocated buffer for cell hidden state
        :param pre_zifo: preallocated buffer that contains precomputed W * x
        :param dL_dpre_zifo: preallocated buffer that contains vertically
                             stacked dL_dpre_z, dL_dpre_i, dL_dpre_f, dL_dpre_o
        :param prev_c: previous lstm cell state
        :param prev_h: previous hidden lstm cell state
        """

        self.context = context
        self.R = R
        self.pre_zifo = pre_zifo
        self.zifo = Matrix.empty_like(pre_zifo, context.device_id)
        dim = h.nrows
        self.z = self.zifo[0*dim:1*dim, 0]
        self.i = self.zifo[1*dim:2*dim, 0]
        self.f = self.zifo[2*dim:3*dim, 0]
        self.o = self.zifo[3*dim:4*dim, 0]
        self.c = Matrix.empty_like(h, context.device_id)
        self.tanh_c = Matrix.empty_like(h, context.device_id)
        self.h = h
        self.prev_c = prev_c
        self.prev_h = prev_h
        self.learning = learning
        if learning:
            self._dzifo_dpre_zifo = Matrix.empty_like(pre_zifo, context.device_id)
            self._dz_dpre_z = self._dzifo_dpre_zifo[0*dim:1*dim, 0]
            self._di_dpre_i = self._dzifo_dpre_zifo[1*dim:2*dim, 0]
            self._df_dpre_f = self._dzifo_dpre_zifo[2*dim:3*dim, 0]
            self._do_dpre_o = self._dzifo_dpre_zifo[3*dim:4*dim, 0]

            self.dL_dpre_zifo = dL_dpre_zifo
            self.dL_dpre_z = self.dL_dpre_zifo[0*dim:1*dim, 0]
            self.dL_dpre_i = self.dL_dpre_zifo[1*dim:2*dim, 0]
            self.dL_dpre_f = self.dL_dpre_zifo[2*dim:3*dim, 0]
            self.dL_dpre_o = self.dL_dpre_zifo[3*dim:4*dim, 0]

            self.dL_dc = Matrix.empty_like(self.c, context.device_id)
            self._dtanh_c_dc = Matrix.empty_like(h, context.device_id)

    @property
    def dzifo_dpre_zifo(self):
        if self.learning:
            return self._dzifo_dpre_zifo

    @property
    def dz_dpre_z(self):
        if self.learning:
            return self._dz_dpre_z

    @property
    def di_dpre_i(self):
        if self.learning:
            return self._di_dpre_i

    @property
    def df_dpre_f(self):
        if self.learning:
            return self._df_dpre_f

    @property
    def do_dpre_o(self):
        if self.learning:
            return self._do_dpre_o

    @property
    def dtanh_c_dc(self):
        if self.learning:
            return self._dtanh_c_dc

    def fprop(self):
        # zifo = tanh_sigm(W * x[t] + R * h[t-1])
        self.pre_zifo.add_dot(self.context, self.R, self.prev_h)
        self.pre_zifo.tanh_sigm(self.context, self.zifo, self.dzifo_dpre_zifo)

        # c[t] = i[t] .* z[t] + f[t] .* c[t-1]
        # h[t] = o[t] .* tanh(c[t])
        self.c.assign_sum_hprod(self.context, self.i, self.z, self.f, self.prev_c)
        self.c.tanh(self.context, self.tanh_c, self.dtanh_c_dc)
        self.h.assign_hprod(self.context, self.o, self.tanh_c)

    def bprop(self, context, dL_dh, dL_dprev_h, dL_dprev_c):
        # dL/dc[t] += dL/dh[t] .* o[t] .* dtanh(c[t])/dc[t]
        self.dL_dc.add_hprod(context, dL_dh, self.o, self.dtanh_c_dc)

        # dL/dpre_o[t] = dL/dh[t] .* tanh(c[t]) .* do[t]/dpre_o[t]
        # dL/dpre_f[t] = dL/dc[t] .* c[t-1] .* df[t]/dpre_f[t]
        # dL/dpre_i[t] = dL/dc[t] .* z[t] .* di[t]/dpre_i[t]
        # dL/dpre_z[t] = dL/dc[t] .* i[t] .* dz[t]/dpre_z[t]
        self.dL_dpre_o.assign_hprod(context, dL_dh, self.tanh_c, self.do_dpre_o)
        self.dL_dpre_f.assign_hprod(context, self.dL_dc, self.prev_c, self.df_dpre_f)
        self.dL_dpre_i.assign_hprod(context, self.dL_dc, self.z, self.di_dpre_i)
        self.dL_dpre_z.assign_hprod(context, self.dL_dc, self.i, self.dz_dpre_z)

        if dL_dprev_h:
            # dL/dh[t-1] = R.T * dL/dpre_zifo[t]
            dL_dprev_h.add_dot(context, self.R, self.dL_dpre_zifo, 'T')
            # dL/dc[t-1] = f[t] .* dL/dc[t]
            dL_dprev_c.assign_hprod(context, self.f, self.dL_dc)