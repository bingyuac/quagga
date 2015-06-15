from quagga.matrix import Matrix
from quagga.blocks import Connector


class NpLstmCell(object):
    def __init__(self, W, R, h, pre_zifo, dL_dpre_zifo, prev_c, prev_h, context, propagate_error=True):
        """
        No peepholes LSTM cell block is used for building `NpLstmRnn` block.
        This block is not completely autonomous it requires precomputed
        `W * x` -- pre_zifo which is not the connector. That is why `NpLstmRnn`
        should take care of proper synchronization.

        :param W: matrix that contains horizontally stacked Wz, Wi, Wf, Wo
        :param R: matrix that contains horizontally stacked Rz, Ri, Rf, Ro
        :param h: preallocated buffer for cell hidden state
        :param pre_zifo: precomputed W * x
        :param dL_dpre_zifo: preallocated buffer that contains horizontally
                             stacked dL_dpre_z, dL_dpre_i, dL_dpre_f, dL_dpre_o
        :param prev_c: connector to previous lstm cell state
        :param prev_h: connector to previous hidden lstm cell state
        :param context: context in which computation occurs
        :param propagate_error:
        :return:
        """

        self.W = W
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
        self.context = context

        self.c = Connector(Matrix.empty_like(h), self.context)
        self.tanh_c = Matrix.empty_like(h)
        self._dtanh_c_dc = Matrix.empty_like(h)
        self.h = Connector(h, self.context)

        self.prev_c = prev_c
        self.prev_h = prev_h
        if propagate_error:
            self.prev_c.register_user(self, self.context)
            self.prev_h.register_user(self, self.context)
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

    def fprop(self):
        # zifo = tanh_sigm(W * x[t] + R * h[t-1])
        self.prev_h.forward_block(self.context)
        self.pre_zifo.add_dot(self.context, self.R, self.prev_h)
        self.pre_zifo.tanh_sigm(self.context, self.zifo, self.dzifo_dpre_zifo)

        # c[t] = i[t] .* z[t] + f[t] .* c[t-1]
        # h[t] = o[t] .* tanh(c[t])
        self.prev_c.block(self.context)
        self.c.assign_sum_hprod(self.context, self.i, self.z, self.f, self.prev_c)
        self.c.tanh(self.context, self.tanh_c, self.dtanh_c_dc)
        self.h.assign_hprod(self.context, self.o, self.tanh_c)

    def bprop(self):
        # dL/dc[t] += dL/dh[t] .* o[t] .* dtanh(c[t])/dc[t]
        self.h.backward_block(self.context)
        self.c.backward_block(self.context)
        self.c.derivative.add_hprod(self.context, self.h.derivative, self.o, self.dtanh_c_dc)

        # dL/dpre_o[t] = dL/dh[t] .* tanh(c[t]) .* do[t]/dpre_o[t]
        # dL/dpre_f[t] = dL/dc[t] .* c[t-1] .* df[t]/dpre_f[t]
        # dL/dpre_i[t] = dL/dc[t] .* z[t] .* di[t]/dpre_i[t]
        # dL/dpre_z[t] = dL/dc[t] .* i[t] .* dz[t]/dpre_z[t]
        self.dL_dpre_o.assign_hprod(self.context, self.h.derivative, self.tanh_c, self.do_dpre_o)
        self.dL_dpre_f.assign_hprod(self.context, self.c.derivative, self.prev_c, self.df_dpre_f)
        self.dL_dpre_i.assign_hprod(self.context, self.c.derivative, self.z, self.di_dpre_i)
        self.dL_dpre_z.assign_hprod(self.context, self.c.derivative, self.i, self.dz_dpre_z)

        if self.propagate_error:
            # dL/dh[t-1] = R.T * dL/dpre_zifo[t]
            self.prev_h.get_derivative(self).assign_dot(self.context, self.R, self.dL_dpre_zifo, 'T')
            # dL/dc[t-1] = f[t] .* dL/dc[t]
            self.prev_c.get_derivative(self).assign_hprod(self.context, self.f, self.c.derivative)