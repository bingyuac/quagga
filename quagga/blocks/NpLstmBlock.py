from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class NpLstmBlock(object):
    def __init__(self, W, R, x, prev_c, prev_h, learning=True, device_id=None):
        """
        TODO

        :param W: matrix that contains vertically stacked Wz, Wi, Wf, Wo
        :param R: matrix that contains vertically stacked Rz, Ri, Rf, Ro
        :param prev_c: previous lstm cell state
        :param prev_h: previous hidden lstm state
        TODO
        """

        self.context = Context(device_id)
        device_id = self.context.device_id
        self.W = W # TODO register usage here
        self.R = R # TODO register usage here
        self.pre_zifo = Matrix.empty_like(prev_c, device_id)
        self.zifo = Matrix.empty_like(prev_c, device_id)
        dim = prev_c.nrows
        self.z = self.zifo[0*dim:1*dim, 0]
        self.i = self.zifo[1*dim:2*dim, 0]
        self.f = self.zifo[2*dim:3*dim, 0]
        self.o = self.zifo[3*dim:4*dim, 0]
        self.c = Matrix.empty_like(prev_c, device_id)
        self.tanh_c = Matrix.empty_like(prev_c, device_id)
        self.h = Matrix.empty_like(prev_c, device_id)
        self.prev_c, self.dL_dprev_c = prev_c.register_usage(self.context, self.context)
        self.prev_h, self.dL_dprev_h = prev_h.register_usage(self.context, self.context)
        self.learning = learning
        if learning:
            self._dzifo_dpre_zifo = Matrix.empty_like(self.pre_zifo, device_id)
            self._dz_dpre_z = self._dzifo_dpre_zifo[0*dim:1*dim, 0]
            self._di_dpre_i = self._dzifo_dpre_zifo[1*dim:2*dim, 0]
            self._df_dpre_f = self._dzifo_dpre_zifo[2*dim:3*dim, 0]
            self._do_dpre_o = self._dzifo_dpre_zifo[3*dim:4*dim, 0]

            self.dL_dpre_zifo = Matrix.empty_like(self.pre_zifo, device_id)
            self.dL_dpre_z = self.dL_dpre_zifo[0*dim:1*dim, 0]
            self.dL_dpre_i = self.dL_dpre_zifo[1*dim:2*dim, 0]
            self.dL_dpre_f = self.dL_dpre_zifo[2*dim:3*dim, 0]
            self.dL_dpre_o = self.dL_dpre_zifo[3*dim:4*dim, 0]

            self.dL_dc = Matrix.empty_like(self.c, device_id)
            self._dtanh_c_dc = Matrix.empty_like(self.c, device_id)

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
        self.pre_zifo.assign_dot(self.context, self.W, self.x)
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