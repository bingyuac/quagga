import ctypes
import numpy as np
from itertools import izip
from network import LstmBlock, MarginalLstmBlock
from network import MatrixClass, MatrixContextClass


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
            for gate in ['z', 'i', 'f', 'o']:
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
            self.dL_dx[d] = Matrix.empty(p_init.nrows, max_input_sequence_len)
            self.dL_dh[d] = Matrix.empty_like(p_init)
            for gate in ['z', 'i', 'f', 'o']:
                self.dL_dW[gate, d] = Matrix.empty_like(W_init)
                self.dL_dR[gate, d] = Matrix.empty_like(R_init)
                if gate != 'z':
                    self.dL_dp[gate, d] = Matrix.empty_like(p_init)
                self.dL_dpre[gate, d] = Matrix.empty(p_init.nrows, max_input_sequence_len)

        Context = MatrixContextClass[self.p_type]
        self.context = {}
        for d in ['forward', 'backward']:
            for gate in ['z', 'i', 'f', 'c', 'o']:
                self.context[gate, d] = Context()

        self.lstm_blocks = {'forward': [],
                            'backward': []}
        for d in self.lstm_blocks:
            for n in xrange(max_input_sequence_len):
                cell = LstmBlock(p_type,
                                 self.W['z', d], self.R['z', d],
                                 self.W['i', d], self.R['i', d], self.p['i', d],
                                 self.W['f', d], self.R['f', d], self.p['f', d],
                                 self.W['o', d], self.R['o', d], self.p['o', d],
                                 self.c[d][:, n], self.h[d][:, n],
                                 self.dL_dpre['z', d][:, n],
                                 self.dL_dpre['i', d][:, n],
                                 self.dL_dpre['f', d][:, n],
                                 self.dL_dpre['o', d][:, n],
                                 self.context['z', d],
                                 self.context['i', d],
                                 self.context['f', d],
                                 self.context['c', d],
                                 self.context['o', d])
                if n == 0:
                    cell.prev_cell = MarginalLstmBlock(p_type, p_init.nrows)
                else:
                    cell.prev_cell = self.lstm_blocks[d][-1]
                    self.lstm_blocks[d][-1].next_cell = cell
                self.lstm_blocks[d].append(cell)

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
            for gate in ['z', 'i', 'f', 'o']:
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
        for i in reversed(xrange(n)):
            if i == n - 1:
                for d in ['forward', 'backward']:
                    # dL/dh = h * error
                    self.lstm_blocks[d][n].h.scale(self.context['o', d], error, self.dL_dh[d])
                    self.lstm_blocks[d][n].backward_propagation(self.dL_dh[d])
            else:
                for d in ['forward', 'backward']:
                    self.lstm_blocks[d][n].backward_propagation()

        # dL/dx = Wz_f.T * dL/dpre_z_f + Wi_f.T * dL/dpre_i_f + Wf_f.T * dL/dpre_f_f + Wo_f.T * dL/dpre_o_f +
        #         Wz_b.T * dL/dpre_z_b + Wi_b.T * dL/dpre_i_b + Wf_b.T * dL/dpre_f_b + Wo_b.T * dL/dpre_o_b
        dL_dpre = {}
        dL_dx = {}
        for d in ['forward', 'backward']:
            dL_dx[d] = self.dL_dx[d][:, :n]
            context = self.context['z', d]
            for gate in ['z', 'i', 'f', 'o']:
                dL_dpre[gate, d] = self.dL_dpre[gate, d][:, :n]
                if gate == 'z':
                    dL_dx[d].assign_dot(context, self.W[gate, d], dL_dpre[gate, d], 'T')
                else:
                    dL_dx[d].add_dot(context, self.W[gate, d], dL_dpre[gate, d], 'T')

        # TODO edit these comments when you end debugging
        # dL_dW@ = dL_dpre_@ * x
        # dL_dR@ = dL_dpre_@ * h
        # dL_dp@ = dL_dpre_@ * c
        for d in ['forward', 'backward']:
            for gate in ['z', 'i', 'f', 'o']:
                self.dL_dW[gate, d].assign_dot(self.context[gate, d], input_sequence, dL_dpre[gate, d], 'T')
                self.dL_dR[gate, d].assign_dot(self.context[gate, d], self.h[:, :n], dL_dpre[gate, d], 'T')
                if gate != 'z':
                    self.dL_dp[gate, d].assign_dot(self.context[gate, d], self.c[:, :n], dL_dpre[gate, d], 'T')

        conts = [v for k, v in self.context.iteritems() if k != ('o', 'backward')]
        cont_o_b = self.context['o', 'backward']
        cont_o_b.depend_on(conts)
        self.dL_dx['forward'].add(cont_o_b, self.dL_dx['backward'])

        return self.dL_dW, self.dL_dR, self.dL_dp, self.dL_dw_hy, self.dL_dx['forward']