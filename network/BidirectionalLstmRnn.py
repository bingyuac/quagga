from network import LstmBlock
from network import MatrixClass, MatrixContextClass


class BidirectionalLstmRnn(object):
    def __init__(self, p_type, max_sentence_len,
                 W_init, R_init, p_init, logistic_init):
        #          c, h, dL_dpre_z, dL_dpre_i, dL_dpre_f, dL_dpre_o,
        #          z_context, i_context, f_context, c_context, o_context
        self.p_type = p_type

        Matrix = MatrixClass[self.p_type]
        self.Wz_f = Matrix.from_npa(W_init())
        self.Rz_f = Matrix.from_npa(R_init())
        self.Wi_f = Matrix.from_npa(W_init())
        self.Ri_f = Matrix.from_npa(R_init())
        self.pi_f = Matrix.from_npa(p_init())
        self.Wf_f = Matrix.from_npa(W_init())
        self.Rf_f = Matrix.from_npa(R_init())
        self.pf_f = Matrix.from_npa(p_init())
        self.Wo_f = Matrix.from_npa(W_init())
        self.Ro_f = Matrix.from_npa(R_init())
        self.po_f = Matrix.from_npa(p_init())
        self.w_hy_f = Matrix.from_npa(logistic_init())

        self.Wz_b = Matrix.from_npa(W_init())
        self.Rz_b = Matrix.from_npa(R_init())
        self.Wi_b = Matrix.from_npa(W_init())
        self.Ri_b = Matrix.from_npa(R_init())
        self.pi_b = Matrix.from_npa(p_init())
        self.Wf_b = Matrix.from_npa(W_init())
        self.Rf_b = Matrix.from_npa(R_init())
        self.pf_b = Matrix.from_npa(p_init())
        self.Wo_b = Matrix.from_npa(W_init())
        self.Ro_b = Matrix.from_npa(R_init())
        self.po_b = Matrix.from_npa(p_init())
        self.w_hy_b = Matrix.from_npa(logistic_init())

        self.pre_z_f = Matrix.empty(p_init.nrows, max_sentence_len)
        self.pre_i_f = Matrix.empty(p_init.nrows, max_sentence_len)
        self.pre_f_f = Matrix.empty(p_init.nrows, max_sentence_len)
        self.c_f = Matrix.empty(p_init.nrows, max_sentence_len)
        self.pre_o_f = Matrix.empty(p_init.nrows, max_sentence_len)
        self.h_f = Matrix.empty(p_init.nrows, max_sentence_len)

        self.pre_z_b = Matrix.empty(p_init.nrows, max_sentence_len)
        self.pre_i_b = Matrix.empty(p_init.nrows, max_sentence_len)
        self.pre_f_b = Matrix.empty(p_init.nrows, max_sentence_len)
        self.c_b = Matrix.empty(p_init.nrows, max_sentence_len)
        self.pre_o_b = Matrix.empty(p_init.nrows, max_sentence_len)
        self.h_b = Matrix.empty(p_init.nrows, max_sentence_len)

        self.dL_dh_f = Matrix.empty_like(p_init)
        self.dL_dh_b = Matrix.empty_like(p_init)

        self.dL_dpre_z_f = Matrix.empty(p_init.nrows, max_sentence_len)
        self.dL_dpre_i_f = Matrix.empty(p_init.nrows, max_sentence_len)
        self.dL_dpre_f_f = Matrix.empty(p_init.nrows, max_sentence_len)
        self.dL_dpre_o_f = Matrix.empty(p_init.nrows, max_sentence_len)

        self.dL_dpre_z_b = Matrix.empty(p_init.nrows, max_sentence_len)
        self.dL_dpre_i_b = Matrix.empty(p_init.nrows, max_sentence_len)
        self.dL_dpre_f_b = Matrix.empty(p_init.nrows, max_sentence_len)
        self.dL_dpre_o_b = Matrix.empty(p_init.nrows, max_sentence_len)

        self.dL_dWz_f = Matrix.empty_like(W_init)
        self.dL_dRz_f = Matrix.empty_like(R_init)
        self.dL_dWi_f = Matrix.empty_like(W_init)
        self.dL_dRi_f = Matrix.empty_like(R_init)
        self.dL_dpi_f = Matrix.empty_like(p_init)
        self.dL_dWf_f = Matrix.empty_like(W_init)
        self.dL_dRf_f = Matrix.empty_like(R_init)
        self.dL_dpf_f = Matrix.empty_like(p_init)
        self.dL_dWo_f = Matrix.empty_like(W_init)
        self.dL_dRo_f = Matrix.empty_like(R_init)
        self.dL_dpo_f = Matrix.empty_like(p_init)
        self.dL_dw_hy_f = Matrix.empty_like(logistic_init)
        self.dL_dx_f = Matrix.empty(p_init.nrows, max_sentence_len)

        self.dL_dWz_b = Matrix.empty_like(W_init)
        self.dL_dRz_b = Matrix.empty_like(R_init)
        self.dL_dWi_b = Matrix.empty_like(W_init)
        self.dL_dRi_b = Matrix.empty_like(R_init)
        self.dL_dpi_b = Matrix.empty_like(p_init)
        self.dL_dWf_b = Matrix.empty_like(W_init)
        self.dL_dRf_b = Matrix.empty_like(R_init)
        self.dL_dpf_b = Matrix.empty_like(p_init)
        self.dL_dWo_b = Matrix.empty_like(W_init)
        self.dL_dRo_b = Matrix.empty_like(R_init)
        self.dL_dpo_b = Matrix.empty_like(p_init)
        self.dL_dw_hy_b = Matrix.empty_like(logistic_init)
        self.dL_dx_b = Matrix.empty(p_init.nrows, max_sentence_len)

        Context = MatrixContextClass[self.p_type]
        self.z_f_context = Context()
        self.i_f_context = Context()
        self.f_f_context = Context()
        self.c_f_context = Context()
        self.o_f_context = Context()

        self.z_b_context = Context()
        self.i_b_context = Context()
        self.f_b_context = Context()
        self.c_b_context = Context()
        self.o_b_context = Context()

        self.forward_lstm_blocks = []
        self.backward_lstm_blocks = []
        for n in xrange(max_sentence_len):
            cell = LstmBlock(p_type, self.Wz_f, self.Rz_f,
                                     self.Wi_f, self.Ri_f, self.pi_f,
                                     self.Wf_f, self.Rf_f, self.pf_f,
                                     self.Wo_f, self.Ro_f, self.po_f,
                                     self.c_f[:, n], self.h_f[:, n],
                                     self.dL_dpre_z_f[:, n],
                                     self.dL_dpre_i_f[:, n],
                                     self.dL_dpre_f_f[:, n],
                                     self.dL_dpre_o_f[:, n],
                                     self.z_f_context,
                                     self.i_f_context,
                                     self.f_f_context,
                                     self.c_f_context,
                                     self.o_f_context, prev_cell=None, next_cell=None)
            self.forward_lstm_blocks.append(cell)