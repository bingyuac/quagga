import numpy as np
from network import MatrixClass


class LstmBlock(object):
    def __init__(self, p_type, Wz, Rz, Wi, Ri, pi, Wf, Rf, pf, Wo, Ro, po,
                 c, h, dL_dpre_z, dL_dpre_i, dL_dpre_f, dL_dpre_o,
                 z_context, i_context, f_context, c_context, o_context):
        self.p_type = p_type
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
        self.z = MatrixClass[self.p_type].empty_like(c)
        self.i = MatrixClass[self.p_type].empty_like(c)
        self.f = MatrixClass[self.p_type].empty_like(c)
        self.c = c
        self.tanh_c = MatrixClass[self.p_type].empty_like(c)
        self.o = MatrixClass[self.p_type].empty_like(c)
        self.h = h
        self._dz_dpre_z = MatrixClass[self.p_type].empty_like(self.z)
        self._di_dpre_i = MatrixClass[self.p_type].empty_like(self.i)
        self._df_dpre_f = MatrixClass[self.p_type].empty_like(self.f)
        self._dtanh_c_dc = MatrixClass[self.p_type].empty_like(self.c)
        self._do_dpre_o = MatrixClass[self.p_type].empty_like(self.o)
        self.dL_dh = MatrixClass[self.p_type].empty_like(self.h)
        self.dL_dhz = MatrixClass[self.p_type].empty_like(self.z)
        self.dL_dhi = MatrixClass[self.p_type].empty_like(self.i)
        self.dL_dhf = MatrixClass[self.p_type].empty_like(self.f)
        self.dL_dc = MatrixClass[self.p_type].empty_like(self.c)
        self.dL_dpre_z = dL_dpre_z
        self.dL_dpre_i = dL_dpre_i
        self.dL_dpre_f = dL_dpre_f
        self.dL_dpre_o = dL_dpre_o
        self.z_context = z_context
        self.i_context = i_context
        self.f_context = f_context
        self.c_context = c_context
        self.o_context = o_context
        self.prev_cell = None
        self.next_cell = None
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

    def forward_propagation(self, pre_z, pre_i, pre_f, pre_o):
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
        MatrixClass[self.p_type].sum_hprod(self.c_context, self.c, self.i, self.z, self.f, self.prev_cell.c)
        self.c.tanh(self.c_context, self.tanh_c, self.dtanh_c_dc)

        # o[t] = sigmoid(Wo * x[t] + Ro * h[t-1] + po .* c[t])
        # h[t] = o[t] .* tanh(c[t])
        pre_o.add_dot(self.o_context, self.Ro, self.prev_cell.h_t)
        self.o_context.depend_on(self.c_context)
        pre_o.add_hprod(self.o_context, self.po, self.c)
        pre_o.sigmoid(self.o_context, self.o, self.do_dpre_o)
        MatrixClass[self.p_type].hprod(self.o_context, self.h, self.o, self.tanh_c)
        self.o_context.block(self.z_context, self.i_context, self.f_context)

    def backward_propagation(self, dL_dh=None):
        if dL_dh:
            # dL/dpre_o[t] = dL/dh[t] .* tanh(c[t]) .* do[t]/dpre_o[t]
            MatrixClass[self.p_type].hprod(self.o_context, self.dL_dpre_o, dL_dh, self.tanh_c, self.do_dpre_o)
            # dL/dc[t] = dL/dh[t] .* o[t] .* dtanh(c[t])/dc[t] + po .* dL/dpre_o[t]
            MatrixClass[self.p_type].sum_hprod(self.o_context, self.dL_dc, dL_dh, self.o, self.dtanh_c_dc, self.po, self.dL_dpre_o)
        else:
            # dL/dh[t] = Rz.T * dL/dpre_z[t+1] +
            #            Ri.T * dL/dpre_i[t+1] +
            #            Rf.T * dL/dpre_f[t+1] +
            #            Ro.T * dL/dpre_o[t+1]
            self.dL_dhz.assign_dot(self.z_context, self.Rz, self.next_cell.dL_dpre_z, matrix_operation='T')
            self.dL_dhi.assign_dot(self.i_context, self.Ri, self.next_cell.dL_dpre_i, matrix_operation='T')
            self.dL_dhf.assign_dot(self.f_context, self.Rf, self.next_cell.dL_dpre_f, matrix_operation='T')
            self.dL_dh.assign_dot(self.o_context, self.Ro, self.next_cell.dL_dpre_o, matrix_operation='T')
            self.o_context.depend_on(self.z_context, self.i_context, self.f_context)
            self.dL_dh.add(self.o_context, self.dL_dhz, self.dL_dhi, self.dL_dhf)

            # dL/dpre_o[t] = dL/dh[t] .* tanh(c[t]) .* do[t]/dpre_o[t]
            MatrixClass[self.p_type].hprod(self.o_context, self.dL_dpre_o, self.dL_dh, self.tanh_c, self.do_dpre_o)

            # dL/dc[t] = dL/dh[t] .* o[t] .* dtanh(c[t])/dc[t] +
            #            po .* dL/dpre_o[t] +
            #            pi .* dL/dpre_i[t+1] +
            #            pf .* dL/dpre_f[t+1] +
            #            dL/dc[t+1] .* f[t+1]
            MatrixClass[self.p_type].sum_hprod(self.o_context, self.dL_dc,
                                               self.dL_dh, self.o, self.dtanh_c_dc,
                                               self.po, self.dL_dpre_o,
                                               self.pi, self.next_cell.dL_dpre_i,
                                               self.pf, self.next_cell.dL_dpre_f,
                                               self.next_cell.dL_dc, self.next_cell.f)

        # dL/dpre_f[t] = dL/dc[t] .* c[t-1] .* df[t]/dpre_f[t]
        # dL/dpre_i[t] = dL/dc[t] .* z[t] .* di[t]/dpre_i[t]
        # dL/dpre_z[t] = dL/dc[t] .* i[t] .* dz[t]/dpre_z[t]
        self.o_context.block(self.z_context, self.i_context, self.f_context)
        MatrixClass[self.p_type].hprod(self.f_context, self.dL_dpre_f, self.dL_dc, self.prev_cell.c, self.df_dpre_f)
        MatrixClass[self.p_type].hprod(self.i_context, self.dL_dpre_i, self.dL_dc, self.z, self.di_dpre_i)
        MatrixClass[self.p_type].hprod(self.z_context, self.dL_dpre_z, self.dL_dc, self.i, self.dz_dpre_z)


class MarginalLstmBlock(object):
    def __init__(self, p_type, n):
        zero_vector = MatrixClass[p_type].from_npa(np.zeros(n, 1))
        self.c = zero_vector
        self.h = zero_vector