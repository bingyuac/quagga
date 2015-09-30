from quagga.matrix import Matrix
from quagga.context import Context
from quagga.connector import Connector


class LstmBlock(object):
    def __init__(self, W, R, x, prev_c, prev_h, device_id, mask=None):
        """
        TODO

        :param W: connector that contains horizontally stacked Wz, Wi, Wf, Wo
        :param R: connector that contains horizontally stacked Rz, Ri, Rf, Ro
        :param prev_c: previous lstm cell state
        :param prev_h: previous lstm hidden state

        TODO
        """

        self.f_context = Context(device_id)
        device_id = self.f_context.device_id
        if isinstance(mask, Connector):
            self.mask = mask.register_usage(device_id)
        else:
            self.mask = mask
        if W.bpropagable:
            self.W, self.dL_dW = W.register_usage(device_id, device_id)
            self.W_b_context = Context(device_id)
        else:
            self.W = W.register_usage(device_id)
        if R.bpropagable:
            self.R, self.dL_dR = R.register_usage(device_id, device_id)
            self.R_b_context = Context(device_id)
        else:
            self.R = R.register_usage(device_id)
        if x.bpropagable:
            self.x, self.dL_dx = x.register_usage(device_id, device_id)
            self.x_b_context = Context(device_id)
        else:
            self.x = x.register_usage(device_id)
        if prev_c.bpropagable:
            self.prev_c, self.dL_dprev_c = prev_c.register_usage(device_id, device_id)
        else:
            self.prev_c = prev_c.register_usage(device_id)
        if prev_h.bpropagable:
            self.prev_h, self.dL_dprev_h = prev_h.register_usage(device_id, device_id)
        else:
            self.prev_h = prev_h.register_usage(device_id)
        self.learning = W.bpropagable or R.bpropagable or x.bpropagable or \
                        prev_c.bpropagable or prev_h.bpropagable
        if self.learning:
            self.b_context = Context(device_id)

        dim = self.R.nrows
        batch_size = self.x.nrows

        self.zifo = Matrix.empty(batch_size, 4 * dim, device_id=device_id)
        self.z = self.zifo[:, 0*dim:1*dim]
        self.i = self.zifo[:, 1*dim:2*dim]
        self.f = self.zifo[:, 2*dim:3*dim]
        self.o = self.zifo[:, 3*dim:4*dim]
        self.c = Matrix.empty_like(self.prev_c, device_id)
        self.c = Connector(self.c, device_id if self.learning else None)
        self.tanh_c = Matrix.empty_like(self.c, device_id)
        self.h = Matrix.empty_like(self.c, device_id)
        self.h = Connector(self.h, device_id if self.learning else None)

        if self.learning:
            self._dzifo_dpre_zifo = Matrix.empty_like(self.zifo)
            self.dz_dpre_z = self._dzifo_dpre_zifo[:, 0*dim:1*dim]
            self.di_dpre_i = self._dzifo_dpre_zifo[:, 1*dim:2*dim]
            self.df_dpre_f = self._dzifo_dpre_zifo[:, 2*dim:3*dim]
            self.do_dpre_o = self._dzifo_dpre_zifo[:, 3*dim:4*dim]
            self.dL_dpre_zifo = self._dzifo_dpre_zifo
            self.dL_dpre_z = self.dz_dpre_z
            self.dL_dpre_i = self.di_dpre_i
            self.dL_dpre_f = self.df_dpre_f
            self.dL_dpre_o = self.do_dpre_o
            self._dtanh_c_dc = Matrix.empty_like(self.c)

    @property
    def dzifo_dpre_zifo(self):
        if self.learning:
            return self._dzifo_dpre_zifo

    @property
    def dtanh_c_dc(self):
        if self.learning:
            return self._dtanh_c_dc

    def fprop(self):
        # zifo = tanh_sigm(x[t] * W + h[t-1] * R)
        self.zifo.assign_dot(self.f_context, self.x, self.W)
        self.zifo.add_dot(self.f_context, self.prev_h, self.R)
        self.zifo.tanh_sigm(self.f_context, self.zifo, self.dzifo_dpre_zifo, axis=1)

        # c[t] = i[t] .* z[t] + f[t] .* c[t-1]
        # h[t] = o[t] .* tanh(c[t])
        self.c.assign_sum_hprod(self.f_context, self.i, self.z, self.f, self.prev_c)
        self.c.tanh(self.f_context, self.tanh_c, self.dtanh_c_dc)
        self.h.assign_hprod(self.f_context, self.o, self.tanh_c)
        if self.mask:
            # s[t] = mask .* s[t] + (1 - mask) .* s[t-1]
            self.c.assign_masked_addition(self.f_context, self.mask, self.c, self.prev_c)
            self.h.assign_masked_addition(self.f_context, self.mask, self.h, self.prev_h)
        self.c.fprop()
        self.h.fprop()

    def bprop(self):
        dL_dc = self.c.backward_matrix
        dL_dh = self.h.backward_matrix
        if self.mask:
            # dL/ds[t-1] = (1 - mask) .* dL/ds[t]
            # dL/ds[t] = mask .* dL/ds[t]
            if hasattr(self, 'dL_dprev_c'):
                self.dL_dprev_c.add_hprod_one_minus_mask(self.b_context, self.mask, dL_dc)
            dL_dc.hprod(self.b_context, self.mask)
            if hasattr(self, 'dL_dprev_h'):
                self.dL_dprev_h.add_hprod_one_minus_mask(self.b_context, self.mask, dL_dh)
            dL_dh.hprod(self.b_context, self.mask)
        # dL/dc[t] = dL[t+1]/dc[t] + dL/dh[t] .* o[t] .* dtanh(c[t])/dc[t]
        dL_dc.add_hprod(self.b_context, dL_dh, self.o, self.dtanh_c_dc)

        # dL/dpre_o[t] = dL/dh[t] .* tanh(c[t]) .* do[t]/dpre_o[t]
        # dL/dpre_f[t] = dL/dc[t] .* c[t-1] .* df[t]/dpre_f[t]
        # dL/dpre_i[t] = dL/dc[t] .* z[t] .* di[t]/dpre_i[t]
        # dL/dpre_z[t] = dL/dc[t] .* i[t] .* dz[t]/dpre_z[t]
        self.dL_dpre_o.assign_hprod(self.b_context, dL_dh, self.tanh_c, self.do_dpre_o)
        self.dL_dpre_f.assign_hprod(self.b_context, dL_dc, self.prev_c, self.df_dpre_f)
        self.dL_dpre_i.assign_hprod(self.b_context, dL_dc, self.z, self.di_dpre_i)
        self.dL_dpre_z.assign_hprod(self.b_context, dL_dc, self.i, self.dz_dpre_z)
        self.dL_dpre_zifo.last_modification_context = self.b_context

        # TODO(sergii): add here clipping of self.dL_dpre_zifo

        if hasattr(self, 'dL_dW'):
            # dL_dW += x[t].T * dL/dpre_zifo[t]
            self.dL_dW.add_dot(self.W_b_context, self.x, self.dL_dpre_zifo, 'T')
        if hasattr(self, 'dL_dR'):
            # dL_dR += h[t-1].T * dL/dpre_zifo[t]
            self.dL_dR.add_dot(self.R_b_context, self.prev_h, self.dL_dpre_zifo, 'T')
        if hasattr(self, 'dL_dx'):
            # dL/dx[t] = dL/dpre_zifo[t] * W.T
            self.dL_dx.add_dot(self.x_b_context, self.dL_dpre_zifo, self.W, 'N', 'T')
        if hasattr(self, 'dL_dprev_c'):
            # dL/dc[t-1] = f[t] .* dL/dc[t]
            self.dL_dprev_c.add_hprod(self.b_context, self.f, dL_dc)
        if hasattr(self, 'dL_dprev_h'):
            # dL/dh[t-1] = dL/dpre_zifo[t] * R.T
            self.dL_dprev_h.add_dot(self.b_context, self.dL_dpre_zifo, self.R, 'N', 'T')