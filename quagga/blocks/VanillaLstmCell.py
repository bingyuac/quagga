from quagga.matrix import Matrix
from quagga.blocks import Connector


class VanillaLstmCell(object):
    def __init__(self, Wz, Rz, Wi, Ri, pi, Wf, Rf, pf, Wo, Ro, po,
                 c, h, dL_dpre_z, dL_dpre_i, dL_dpre_f, dL_dpre_o,
                 z_context, i_context, f_context, o_context):
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
        self.c = Connector(c, self.o_context)
        self.tanh_c = Matrix.empty_like(po)
        self.o = Matrix.empty_like(po)
        self.h = Connector(h, self.o_context)
        self._dz_dpre_z = Matrix.empty_like(self.z)
        self._di_dpre_i = Matrix.empty_like(self.i)
        self._df_dpre_f = Matrix.empty_like(self.f)
        self._dtanh_c_dc = Matrix.empty_like(self.c)
        self._do_dpre_o = Matrix.empty_like(self.o)
        self.dL_dpre_z = dL_dpre_z
        self.dL_dpre_i = dL_dpre_i
        self.dL_dpre_f = dL_dpre_f
        self.dL_dpre_o = dL_dpre_o
        self.z_context = z_context
        self.i_context = i_context
        self.f_context = f_context
        self.o_context = o_context
        self.prev_c = None
        self.prev_h = None
        self.back_prop = False
        self.propagate_error = False
        self.dL_dhz = None
        self.dL_dhi = None
        self.dL_dhf = None

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
        self.prev_h.block(self.z_context)
        pre_z.add_dot(self.z_context, self.Rz, self.prev_h)
        pre_z.tanh(self.z_context, self.z, self.dz_dpre_z)

        # i[t] = sigmoid(Wi * x[t] + Ri * h[t-1] + pi .* c[t-1])
        self.prev_h.block(self.i_context)
        self.prev_c.block(self.i_context)
        pre_i.add_dot(self.i_context, self.Ri, self.prev_h)
        pre_i.add_hprod(self.i_context, self.pi, self.prev_c)
        pre_i.sigmoid(self.i_context, self.i, self.di_dpre_i)

        # f[t] = sigmoid(Wf * x[t] + Rf * h[t-1] + pf .* c[t-1])
        self.prev_h.block(self.f_context)
        self.prev_c.block(self.f_context)
        pre_f.add_dot(self.f_context, self.Rf, self.prev_h)
        pre_f.add_hprod(self.f_context, self.pf, self.prev_c)
        pre_f.sigmoid(self.f_context, self.f, self.df_dpre_f)

        # c[t] = i[t] .* z[t] + f[t] .* c[t-1]
        # tanh(c[t])
        self.o_context.wait(self.z_context, self.i_context, self.f_context)
        self.c.assign_sum_hprod(self.o_context, self.i, self.z, self.f, self.prev_c)
        self.c.tanh(self.o_context, self.tanh_c, self.dtanh_c_dc)

        # o[t] = sigmoid(Wo * x[t] + Ro * h[t-1] + po .* c[t])
        # h[t] = o[t] .* tanh(c[t])
        pre_o.add_dot(self.o_context, self.Ro, self.prev_h)
        pre_o.add_hprod(self.o_context, self.po, self.c)
        pre_o.sigmoid(self.o_context, self.o, self.do_dpre_o)
        self.h.assign_hprod(self.o_context, self.o, self.tanh_c)

    def bprop(self):
        # dL/dpre_o[ t] = dL/dh[t] .* tanh(c[t]) .* do[t]/dpre_o[t]
        self.h.backward_block(self.o_context)
        self.dL_dpre_o.assign_hprod(self.o_context, self.h.derivative, self.tanh_c, self.do_dpre_o)

        # dL/dc[t] += dL/dh[t] .* o[t] .* dtanh(c[t])/dc[t]
        self.c.backward_block(self.o_context)
        self.c.derivative.add_hprod(self.o_context, self.h.derivative, self.o, self.dtanh_c_dc)

        # dL/dpre_f[t] = dL/dc[t] .* c[t-1] .* df[t]/dpre_f[t]
        # dL/dpre_i[t] = dL/dc[t] .* z[t] .* di[t]/dpre_i[t]
        # dL/dpre_z[t] = dL/dc[t] .* i[t] .* dz[t]/dpre_z[t]
        self.o_context.block(self.z_context, self.i_context, self.f_context)
        self.dL_dpre_f.assign_hprod(self.f_context, self.c.derivative, self.prev_c, self.df_dpre_f)
        self.dL_dpre_i.assign_hprod(self.i_context, self.c.derivative, self.z, self.di_dpre_i)
        self.dL_dpre_z.assign_hprod(self.z_context, self.c.derivative, self.i, self.dz_dpre_z)

        if not self.propagate_error:
            return
        # dL/dh[t-1] = Rz.T * dL/dpre_z[t] + Ri.T * dL/dpre_i[t] +
        #              Rf.T * dL/dpre_f[t] + Ro.T * dL/dpre_o[t]
        self.dL_dhz.assign_dot(self.z_context, self.Rz, self.dL_dpre_z, 'T')
        self.dL_dhi.assign_dot(self.i_context, self.Ri, self.dL_dpre_i, 'T')
        self.dL_dhf.assign_dot(self.f_context, self.Rf, self.dL_dpre_f, 'T')
        dL_dprev_h = self.prev_h.get_derivative(self)
        dL_dprev_h.assign_dot(self.o_context, self.Ro, self.dL_dpre_o, 'T')
        self.o_context.wait(self.z_context, self.i_context, self.f_context)
        dL_dprev_h.add(self.o_context, self.dL_dhz, self.dL_dhi, self.dL_dhf)

        # dL/dc[t-1] = pi .* dL/dpre_i[t] + pf .* dL/dpre_f[t] + f[t] .* dL/dc[t]
        self.prev_c.get_derivative(self).\
            assign_sum_hprod(self.z_context,
                             self.pi, self.dL_dpre_i,
                             self.pf, self.dL_dpre_f,
                             self.f, self.c.derivative)

    def register_inputs(self, prev_c, prev_h, propagate_error=True):
        self.prev_c = prev_c
        self.prev_h = prev_h
        if propagate_error:
            prev_c.register_user(self, self.z_context)
            prev_h.register_user(self, self.o_context)
        self.propagate_error = propagate_error
        self.dL_dhz = Matrix.empty_like(self.h)
        self.dL_dhi = Matrix.empty_like(self.h)
        self.dL_dhf = Matrix.empty_like(self.h)