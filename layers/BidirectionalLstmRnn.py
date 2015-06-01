import numpy as np
from itertools import izip
from layers import LstmBlock, MarginalLstmBlock
from layers import MatrixClass, MatrixContextClass


class BidirectionalLstmRnn(object):
    def __init__(self, p_type, max_input_sequence_len,
                 W_init, R_init, p_init, logistic_init):
        self.p_type = p_type
        self.max_input_sequence_len = max_input_sequence_len

        Matrix = MatrixClass[p_type]
        self.W = {}
        self.R = {}
        self.p = {}
        self.w_hy = {}
        self.pre = {}
        self.pre_columns = {}
        self.c = {}
        self.h = {}
        for d in ['forward', 'backward']:
            for gate in 'zifo':
                self.W[gate, d] = Matrix.from_npa(W_init())
                self.R[gate, d] = Matrix.from_npa(R_init())
                self.p[gate, d] = Matrix.from_npa(p_init())
                self.pre[gate, d] = Matrix.empty(p_init.nrows, max_input_sequence_len)
                self.pre_columns[gate, d] = self.pre[gate, d].to_list()
            self.w_hy[d] = Matrix.from_npa(logistic_init())
            self.c[d] = Matrix.empty(p_init.nrows, max_input_sequence_len)
            self.h[d] = Matrix.empty(p_init.nrows, max_input_sequence_len)

        self.dL_dW = {}
        self.dL_dR = {}
        self.dL_dp = {}
        self.dL_dw_hy = {}
        self.dL_dx = {}
        self.dL_dh = {}
        self.dL_dpre = {}
        for d in ['forward', 'backward']:
            self.dL_dw_hy[d] = Matrix.empty_like(logistic_init)
            self.dL_dx[d] = Matrix.empty(W_init.ncols, max_input_sequence_len)
            self.dL_dh[d] = Matrix.empty_like(p_init)
            for gate in 'zifo':
                self.dL_dW[gate, d] = Matrix.empty_like(W_init)
                self.dL_dR[gate, d] = Matrix.empty_like(R_init)
                if gate != 'z':
                    self.dL_dp[gate, d] = Matrix.empty_like(p_init)
                self.dL_dpre[gate, d] = Matrix.empty(p_init.nrows, max_input_sequence_len)

        Context = MatrixContextClass[self.p_type]
        self.context = {}
        for d in ['forward', 'backward']:
            for gate in 'zifco':
                self.context[gate, d] = Context()

        self.lstm_blocks = {'forward': [],
                            'backward': []}
        for d in self.lstm_blocks:
            for k in xrange(max_input_sequence_len):
                cell = LstmBlock(p_type,
                                 self.W['z', d], self.R['z', d],
                                 self.W['i', d], self.R['i', d], self.p['i', d],
                                 self.W['f', d], self.R['f', d], self.p['f', d],
                                 self.W['o', d], self.R['o', d], self.p['o', d],
                                 self.c[d][:, k], self.h[d][:, k],
                                 self.dL_dpre['z', d][:, k],
                                 self.dL_dpre['i', d][:, k],
                                 self.dL_dpre['f', d][:, k],
                                 self.dL_dpre['o', d][:, k],
                                 self.context['z', d],
                                 self.context['i', d],
                                 self.context['f', d],
                                 self.context['c', d],
                                 self.context['o', d])
                if k == 0:
                    cell.prev_cell = MarginalLstmBlock(p_type, p_init.nrows)
                else:
                    cell.prev_cell = self.lstm_blocks[d][-1]
                    self.lstm_blocks[d][-1].next_cell = cell
                self.lstm_blocks[d].append(cell)

    def synchronize(self):
        for context in self.context.itervalues():
            context.synchronize()

    def set_training_mode(self):
        for d in ['forward', 'backward']:
            for cell in self.lstm_blocks[d]:
                cell.back_prop = True

    def set_testing_mode(self):
        for d in ['forward', 'backward']:
            for cell in self.lstm_blocks[d]:
                cell.back_prop = False

    def forward_propagation(self, input_sequence):
        n = input_sequence.ncols
        if n > self.max_input_sequence_len:
            raise ValueError('Sequence has length: {} that is to long. '
                             'The maximum is: {}'.
                             format(n, self.max_input_sequence_len))
        pre = {}
        for d in ['forward', 'backward']:
            for gate in 'zifo':
                pre[gate, d] = self.pre[gate, d][:, :n]
                pre[gate, d].assign_dot(self.context[gate, d], self.W[gate, d], input_sequence)

        for i in xrange(n):
            for d, t in izip(['forward', 'backward'], [i, n-1-i]):
                pre_z = self.pre_columns['z', d][t]
                pre_i = self.pre_columns['i', d][t]
                pre_f = self.pre_columns['f', d][t]
                pre_o = self.pre_columns['o', d][t]
                cell = self.lstm_blocks[d][i]
                cell.forward_propagation(pre_z, pre_i, pre_f, pre_o)

        pre_exp = 0.0
        for d in ['forward', 'backward']:
            pre_exp += self.lstm_blocks[d][n-1].h.vdot(self.context['o', d], self.w_hy[d]).value
        return 1.0 / (1.0 + np.exp(-pre_exp))

    def backward_propagation(self, input_sequence, sequence_grammaticality):
        predicted_sequence_grammaticality = self.forward_propagation(input_sequence)
        error = sequence_grammaticality - predicted_sequence_grammaticality
        n = input_sequence.ncols

        # dL/dw_hy = h * error
        # dL/h = w_hy * error
        for d in ['forward', 'backward']:
            self.w_hy[d].scale(self.context['o', d], error, self.dL_dh[d])
            self.lstm_blocks[d][n-1].h.scale(self.context['f', d], error, self.dL_dw_hy[d])
            self.lstm_blocks[d][n-1].backward_propagation(self.dL_dh[d])
            for k in reversed(xrange(n-1)):
                self.lstm_blocks[d][k].backward_propagation()

        # dL/dx = Wz_f.T * dL/dpre_z_f + Wi_f.T * dL/dpre_i_f + Wf_f.T * dL/dpre_f_f + Wo_f.T * dL/dpre_o_f +
        #         Wz_b.T * dL/dpre_z_b + Wi_b.T * dL/dpre_i_b + Wf_b.T * dL/dpre_f_b + Wo_b.T * dL/dpre_o_b
        dL_dx = {}
        for d in ['forward', 'backward']:
            dL_dx[d] = self.dL_dx[d][:, :n]
            context = self.context['z', d]
            dL_dx[d].assign_dot(context, self.W['z', d], self.dL_dpre['z', d][:, :n], 'T')
            for gate in 'ifo':
                dL_dx[d].add_dot(context, self.W[gate, d], self.dL_dpre[gate, d][:, :n], 'T')
        self.context['z', 'backward'].depend_on(self.context['z', 'forward'])
        dL_dx['forward'].add(self.context['z', 'backward'], dL_dx['backward'])
        dL_dx = dL_dx['forward']

        # dL_dW@ = dL_dpre_@ * x.T
        # dL_dR@ = dL_dpre_@[:, 1:n] * h[:, :n-1].T
        # dL_dpi = sum(dL_dpre_i[:, 1:n] * c[:, :n-1], axis=1)
        # dL_dpf = sum(dL_dpre_f[:, 1:n] * c[:, :n-1], axis=1)
        # dL_dpo = sum(dL_dpre_o * c, axis=1)
        for d in ['forward', 'backward']:
            for gate in 'zifo':
                self.dL_dW[gate, d].assign_dot(self.context[gate, d], self.dL_dpre[gate, d][:, :n], input_sequence, 'N', 'T')
                self.dL_dR[gate, d].assign_dot(self.context[gate, d], self.dL_dpre[gate, d][:, 1:n], self.h[d][:, :n-1], 'N', 'T')
            for gate in 'if':
                self.dL_dp[gate, d].assign_hprod_sum(self.context[gate, d], self.c[d][:, :n-1], self.dL_dpre[gate, d][:, 1:n])
            self.dL_dp['o', d].assign_hprod_sum(self.context['o', d], self.c[d][:, :n], self.dL_dpre['o', d][:, :n])

        return error, self.dL_dW, self.dL_dR, self.dL_dp, self.dL_dw_hy, dL_dx