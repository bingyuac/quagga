import numpy as np
from quagga.matrix import Matrix, MatrixContext


class LstmBlock(object):
    def __init__(self, Wz, Rz, Wi, Ri, pi, Wf, Rf, pf, Wo, Ro, po,
                 z_context=None, i_context=None, f_context=None, c_context=None, o_context=None,
                 c=None, h=None,
                 dL_dpre_z=None, dL_dpre_i=None, dL_dpre_f=None, dL_dpre_o=None,
                 prev_cell=None, next_cell=None):
        self.Wz = Wz
        self.Rz = Rz
        self.Wi = Wi
        self.Ri = Ri
        self.pi = pi
        self.Wf = Wf
        self.Rf = Rf
        self.pf = pf
        self.Wo = Wo
        self.Ro = Ro
        self.po = po
        self.z = Matrix.empty_like(po)
        self.i = Matrix.empty_like(po)
        self.f = Matrix.empty_like(po)
        self.c = c if c else Matrix.empty_like(po)
        self.tanh_c = Matrix.empty_like(po)
        self.o = Matrix.empty_like(po)
        self.h = h if h else Matrix.empty_like(po)
        self._dz_dpre_z = Matrix.empty_like(self.z)
        self._di_dpre_i = Matrix.empty_like(self.i)
        self._df_dpre_f = Matrix.empty_like(self.f)
        self._dtanh_c_dc = Matrix.empty_like(self.c)
        self._do_dpre_o = Matrix.empty_like(self.o)
        self.dL_dh = Matrix.empty_like(self.h)
        self.dL_dhz = Matrix.empty_like(self.h)
        self.dL_dhi = Matrix.empty_like(self.h)
        self.dL_dhf = Matrix.empty_like(self.h)
        self.dL_dc = Matrix.empty_like(self.c)
        self.dL_dpre_z = dL_dpre_z if dL_dpre_z else Matrix.empty_like(self.z)
        self.dL_dpre_i = dL_dpre_i if dL_dpre_i else Matrix.empty_like(self.i)
        self.dL_dpre_f = dL_dpre_f if dL_dpre_f else Matrix.empty_like(self.f)
        self.dL_dpre_o = dL_dpre_o if dL_dpre_o else Matrix.empty_like(self.o)
        self.z_context = z_context if z_context else MatrixContext()
        self.i_context = i_context if i_context else MatrixContext()
        self.f_context = f_context if f_context else MatrixContext()
        self.c_context = c_context if c_context else MatrixContext()
        self.o_context = o_context if o_context else MatrixContext()
        self.prev_cell = prev_cell
        self.next_cell = next_cell
        self.back_prop = None

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
    def dtanh_c_dc(self):
        if self.back_prop:
            return self._dtanh_c_dc

    @property
    def do_dpre_o(self):
        if self.back_prop:
            return self._do_dpre_o

    def set_training_mode(self):
        self.back_prop = True

    def set_testing_mode(self):
        self.back_prop = False

    def fprop(self, pre_z, pre_i, pre_f, pre_o):
        # z[t] = tanh(Wz * x[t] + Rz * h[t-1])
        pre_z.add_dot(self.z_context, self.Rz, self.prev_cell.h)
        pre_z.tanh(self.z_context, self.z, self.dz_dpre_z)

        # i[t] = sigmoid(Wi * x[t] + Ri * h[t-1] + pi .* c[t-1])
        pre_i.add_dot(self.i_context, self.Ri, self.prev_cell.h)
        pre_i.add_hprod(self.i_context, self.pi, self.prev_cell.c)
        pre_i.sigmoid(self.i_context, self.i, self.di_dpre_i)

        # f[t] = sigmoid(Wf * x[t] + Rf * h[t-1] + pf .* c[t-1])
        pre_f.add_dot(self.f_context, self.Rf, self.prev_cell.h)
        pre_f.add_hprod(self.f_context, self.pf, self.prev_cell.c)
        pre_f.sigmoid(self.f_context, self.f, self.df_dpre_f)

        # c[t] = i[t] .* z[t] + f[t] .* c[t-1]
        # tanh(c[t])
        self.c_context.depend_on(self.z_context, self.i_context, self.f_context)
        self.c.assign_sum_hprod(self.c_context, self.i, self.z, self.f, self.prev_cell.c)
        self.c.tanh(self.c_context, self.tanh_c, self.dtanh_c_dc)

        # o[t] = sigmoid(Wo * x[t] + Ro * h[t-1] + po .* c[t])
        # h[t] = o[t] .* tanh(c[t])
        pre_o.add_dot(self.o_context, self.Ro, self.prev_cell.h)
        self.o_context.depend_on(self.c_context)
        pre_o.add_hprod(self.o_context, self.po, self.c)
        pre_o.sigmoid(self.o_context, self.o, self.do_dpre_o)
        self.h.assign_hprod(self.o_context, self.o, self.tanh_c)
        self.o_context.block(self.z_context, self.i_context, self.f_context)

    def bprop(self, dL_dh=None):
        # TODO rewrite this dL_dh
        if dL_dh:
            # dL/dpre_o[t] = dL/dh[t] .* tanh(c[t]) .* do[t]/dpre_o[t]
            self.dL_dpre_o.assign_hprod(self.o_context, dL_dh, self.tanh_c, self.do_dpre_o)
            # dL/dc[t] = dL/dh[t] .* o[t] .* dtanh(c[t])/dc[t] + po .* dL/dpre_o[t]
            self.dL_dc.assign_sum_hprod(self.o_context, dL_dh, self.o, self.dtanh_c_dc, self.po, self.dL_dpre_o)
        else:
            # dL/dh[t] = Rz.T * dL/dpre_z[t+1] +
            #            Ri.T * dL/dpre_i[t+1] +
            #            Rf.T * dL/dpre_f[t+1] +
            #            Ro.T * dL/dpre_o[t+1]
            self.dL_dhz.assign_dot(self.z_context, self.Rz, self.next_cell.dL_dpre_z, 'T')
            self.dL_dhi.assign_dot(self.i_context, self.Ri, self.next_cell.dL_dpre_i, 'T')
            self.dL_dhf.assign_dot(self.f_context, self.Rf, self.next_cell.dL_dpre_f, 'T')
            self.dL_dh.assign_dot(self.o_context, self.Ro, self.next_cell.dL_dpre_o, 'T')
            self.o_context.depend_on(self.z_context, self.i_context, self.f_context)
            self.dL_dh.add(self.o_context, self.dL_dhz, self.dL_dhi, self.dL_dhf)

            # dL/dpre_o[t] = dL/dh[t] .* tanh(c[t]) .* do[t]/dpre_o[t]
            self.dL_dpre_o.assign_hprod(self.o_context, self.dL_dh, self.tanh_c, self.do_dpre_o)

            # dL/dc[t] = dL/dh[t] .* o[t] .* dtanh(c[t])/dc[t] +
            #            po .* dL/dpre_o[t] +
            #            pi .* dL/dpre_i[t+1] +
            #            pf .* dL/dpre_f[t+1] +
            #            dL/dc[t+1] .* f[t+1]
            self.dL_dc.assign_sum_hprod(self.o_context,
                                        self.dL_dh, self.o, self.dtanh_c_dc,
                                        self.po, self.dL_dpre_o,
                                        self.pi, self.next_cell.dL_dpre_i,
                                        self.pf, self.next_cell.dL_dpre_f,
                                        self.next_cell.dL_dc, self.next_cell.f)

        # dL/dpre_f[t] = dL/dc[t] .* c[t-1] .* df[t]/dpre_f[t]
        # dL/dpre_i[t] = dL/dc[t] .* z[t] .* di[t]/dpre_i[t]
        # dL/dpre_z[t] = dL/dc[t] .* i[t] .* dz[t]/dpre_z[t]
        self.o_context.block(self.z_context, self.i_context, self.f_context)
        self.dL_dpre_f.assign_hprod(self.f_context, self.dL_dc, self.prev_cell.c, self.df_dpre_f)
        self.dL_dpre_i.assign_hprod(self.i_context, self.dL_dc, self.z, self.di_dpre_i)
        self.dL_dpre_z.assign_hprod(self.z_context, self.dL_dc, self.i, self.dz_dpre_z)


class MarginalLstmBlock(object):
    def __init__(self, n):
        zero_vector = Matrix.from_npa(np.zeros((n, 1)))
        self.c = zero_vector
        self.h = zero_vector
        self.dL_dpre_z = None
        self.dL_dpre_i = None
        self.dL_dpre_f = None
        self.dL_dpre_o = None
        self.dL_dc = None
        self.f = None

    def depend_on(self, *args):
        pass

    def block(self, *args):
        pass