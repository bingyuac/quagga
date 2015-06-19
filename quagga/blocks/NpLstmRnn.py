import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import Connector


class NpLstmRnn(object):
    def __init__(self, W_init, R_init, x, max_input_sequence_len, propagate_error=True):
        if W_init.nrows != R_init.nrows:
            raise ValueError('W and R have to have the same number of rows!')
        if R_init.nrows != R_init.ncols:
            raise ValueError('R must be a square matrix!')

        nrows = R_init.nrows
        self.context = Context()
        self.max_input_sequence_len = max_input_sequence_len

        self.W = Matrix.empty(4 * nrows, W_init.ncols, 'float')
        self.W.assign_hstack(self.context, W_init(), W_init(), W_init(), W_init())
        self.dL_dW = Matrix.empty_like(self.W)

        self.R = Matrix.empty(4 * nrows, R_init.ncols, 'float')
        self.R.assign_hstack(self.context, R_init(), R_init(), R_init(), R_init())
        self.dL_dR = Matrix.empty_like(self.R)
        self.context.synchronize()

        self.x = x
        if propagate_error:
            self.dL_dx = Matrix.empty(W_init.ncols, max_input_sequence_len, 'float')
            self.x.register_user(self, self.context, self.dL_dx)
        self.propagate_error = propagate_error

        self.h = Connector(Matrix.empty(nrows, max_input_sequence_len, 'float'), self.context)
        self.pre_zifo = Matrix.empty(4 * nrows, max_input_sequence_len, 'float')
        self.dL_dpre_zifo = Matrix.empty_like(self.pre_zifo)

        self.lstm_cells = []
        for k in xrange(max_input_sequence_len):
            if k == 0:
                prev_c = Matrix.from_npa(np.zeros((nrows, 1)))
                prev_h = prev_c
                propagate_error = False
            else:
                prev_c = self.lstm_cells[-1].c
                prev_h = self.lstm_cells[-1].h
                propagate_error = True
            cell = _NpLstmCell(self.R, self.h[:, k], self.pre_zifo[:, k], self.dL_dpre_zifo[:, k], prev_c, prev_h, self.context, propagate_error)
            self.lstm_cells.append(cell)

    def set_training_mode(self):
        for cell in self.lstm_cells:
            cell.back_prop = True

    def set_testing_mode(self):
        for cell in self.lstm_cells:
            cell.back_prop = False

    def fprop(self):
        n = self.x.ncols
        if n > self.max_input_sequence_len:
            raise ValueError('Sequence has length: {} that is too long. '
                             'The maximum is: {}'.
                             format(n, self.max_input_sequence_len))
        self.h.ncols = n
        self.pre_zifo.ncols = n
        self.dL_dpre_zifo.ncols = n
        if self.propagate_error:
            self.dL_dx.ncols = n

        self.pre_zifo.assign_dot(self.context, self.W, self.x)
        for k in xrange(n):
            self.lstm_cells[k].fprop(self.context)

    def bprop(self):
        self.h.backward_block(self.context)
        n = self.x.ncols
        for k in reversed(xrange(n-1)):
            if k == 0:
                dL_dprev_h = None
                dL_dprev_c = None
            else:
                dL_dprev_h = self.lstm_cells[k-1].dL_dh
                dL_dprev_c = self.lstm_cells[k-1].dL_dc
            self.lstm_cells[k].bprop(self.context, dL_dprev_h, dL_dprev_c)

        # dL/dx = W.T * dL/dpre_zifo
        # dL_dW = dL/dpre_zifo * x.T
        # dL_dR = dL/dpre_zifo[:, 1:n] * h[:, :n-1].T
        if self.propagate_error:
            self.dL_dx.assign_dot(self.context, self.W, self.dL_dpre_zifo, 'T')
        self.dL_dW.assign_dot(self.context, self.dL_dpre_zifo, self.x, 'N', 'T')
        self.dL_dR.assign_dot(self.context, self.dL_dpre_zifo[:, 1:n], self.h[:, :n-1], 'N', 'T')

    @property
    def params(self):
        return [self.W, self.R]

    @property
    def grads(self):
        return [self.dL_dW, self.dL_dR]


class _NpLstmCell(object):
    def __init__(self, R, h, pre_zifo, dL_dpre_zifo, prev_c, prev_h, context, propagate_error=True):
        """
        No peepholes LSTM cell block is used for building `NpLstmRnn` block.
        This block is not completely autonomous it requires precomputed
        `W * x` -- pre_zifo which is not the connector. That is why `NpLstmRnn`
        should take care of proper synchronization.

        :param R: matrix that contains horizontally stacked Rz, Ri, Rf, Ro
        :param h: preallocated buffer for cell hidden state
        :param pre_zifo: preallocated buffer that contains precomputed W * x
        :param dL_dpre_zifo: preallocated buffer that contains horizontally
                             stacked dL_dpre_z, dL_dpre_i, dL_dpre_f, dL_dpre_o
        :param prev_c: previous lstm cell state
        :param prev_h: previous hidden lstm cell state
        :param propagate_error:
        """

        self.R = R
        self.pre_zifo = pre_zifo
        self.zifo = Matrix.empty_like(pre_zifo)
        dim = self.c.nrows
        self.z = self.zifo[0*dim:1*dim, :]
        self.i = self.zifo[1*dim:2*dim, :]
        self.f = self.zifo[2*dim:3*dim, :]
        self.o = self.zifo[3*dim:4*dim, :]
        self._dzifo_dpre_zifo = Matrix.empty_like(pre_zifo)
        self._dz_dpre_z = self._dzifo_dpre_zifo[0*dim:1*dim, :]
        self._di_dpre_i = self._dzifo_dpre_zifo[1*dim:2*dim, :]
        self._df_dpre_f = self._dzifo_dpre_zifo[2*dim:3*dim, :]
        self._do_dpre_o = self._dzifo_dpre_zifo[3*dim:4*dim, :]
        self.dL_dpre_zifo = dL_dpre_zifo
        self.dL_dpre_z = self.dL_dpre_zifo[0*dim:1*dim, :]
        self.dL_dpre_i = self.dL_dpre_zifo[1*dim:2*dim, :]
        self.dL_dpre_f = self.dL_dpre_zifo[2*dim:3*dim, :]
        self.dL_dpre_o = self.dL_dpre_zifo[3*dim:4*dim, :]

        self.c = Matrix.empty_like(h)
        self.dL_dc = Matrix.empty_like(self.c)
        self.tanh_c = Matrix.empty_like(h)
        self._dtanh_c_dc = Matrix.empty_like(h)
        self.h = h
        self.dL_dh = Matrix.empty_like(self.h)

        self.prev_c = prev_c
        self.prev_h = prev_h
        self.propagate_error = propagate_error
        self.back_prop = False

    @property
    def dzifo_dpre_zifo(self):
        if self.back_prop:
            return self._dzifo_dpre_zifo

    @property
    def dz_dpre_z(self):
        if self.back_prop:
            return self._dz_dpre_z

    @property
    def di_dpre_i(self):
        if self.back_prop:
            return self._di_dpre_i

    @property
    def df_dpre_f(self):
        if self.back_prop:
            return self._df_dpre_f

    @property
    def do_dpre_o(self):
        if self.back_prop:
            return self._do_dpre_o

    @property
    def dtanh_c_dc(self):
        if self.back_prop:
            return self._dtanh_c_dc

    def set_training_mode(self):
        self.back_prop = True

    def set_testing_mode(self):
        self.back_prop = False

    def fprop(self, context):
        # zifo = tanh_sigm(W * x[t] + R * h[t-1])
        self.pre_zifo.add_dot(context, self.R, self.prev_h)
        self.pre_zifo.tanh_sigm(context, self.zifo, self.dzifo_dpre_zifo)

        # c[t] = i[t] .* z[t] + f[t] .* c[t-1]
        # h[t] = o[t] .* tanh(c[t])
        self.prev_c.block(context)
        self.c.assign_sum_hprod(context, self.i, self.z, self.f, self.prev_c)
        self.c.tanh(context, self.tanh_c, self.dtanh_c_dc)
        self.h.assign_hprod(context, self.o, self.tanh_c)

    def bprop(self, context, dL_dprev_h, dL_dprev_c):
        # dL/dc[t] += dL/dh[t] .* o[t] .* dtanh(c[t])/dc[t]
        self.dL_dc.add_hprod(context, self.dL_dh, self.o, self.dtanh_c_dc)

        # dL/dpre_o[t] = dL/dh[t] .* tanh(c[t]) .* do[t]/dpre_o[t]
        # dL/dpre_f[t] = dL/dc[t] .* c[t-1] .* df[t]/dpre_f[t]
        # dL/dpre_i[t] = dL/dc[t] .* z[t] .* di[t]/dpre_i[t]
        # dL/dpre_z[t] = dL/dc[t] .* i[t] .* dz[t]/dpre_z[t]
        self.dL_dpre_o.assign_hprod(context, self.dL_dh, self.tanh_c, self.do_dpre_o)
        self.dL_dpre_f.assign_hprod(context, self.dL_dc, self.prev_c, self.df_dpre_f)
        self.dL_dpre_i.assign_hprod(context, self.dL_dc, self.z, self.di_dpre_i)
        self.dL_dpre_z.assign_hprod(context, self.dL_dc, self.i, self.dz_dpre_z)

        if self.propagate_error:
            # dL/dh[t-1] = R.T * dL/dpre_zifo[t]
            dL_dprev_h.assign_dot(context, self.R, self.dL_dpre_zifo, 'T')
            # dL/dc[t-1] = f[t] .* dL/dc[t]
            dL_dprev_c.assign_hprod(context, self.f, self.dL_dc)