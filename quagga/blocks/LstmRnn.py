import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import LstmCell, Connector


class LstmRnn(object):
    def __init__(self, W_init, R_init, p_init, input_sequence, max_input_sequence_len):
        self.max_input_sequence_len = max_input_sequence_len
        self.W, self.dL_dW = {}, {}
        self.R, self.dL_dR = {}, {}
        self.p, self.dL_dp = {}, {}
        self.pre, self.dL_dpre = {}, {}
        self.pre_columns = {}
        self.context = {}
        for node in 'zifo':
            self.W[node] = Matrix.from_npa(W_init())
            self.dL_dW[node] = Matrix.empty_like(W_init)
            self.R[node] = Matrix.from_npa(R_init())
            self.dL_dR[node] = Matrix.empty_like(R_init)
            if node != 'z':
                self.p[node] = Matrix.from_npa(p_init())
                self.dL_dp[node] = Matrix.empty_like(p_init)
            self.pre[node] = Matrix.empty(p_init.nrows, max_input_sequence_len)
            self.dL_dpre[node] = Matrix.empty(p_init.nrows, max_input_sequence_len)
            self.pre_columns[node] = self.pre[node].to_list()
            self.context[node] = Context()
        self.c = Matrix.empty(p_init.nrows, max_input_sequence_len)
        self.h = Matrix.empty(p_init.nrows, max_input_sequence_len)
        # self.dL_dx = Matrix.empty(W_init.ncols, max_input_sequence_len)
        self.dL_dx = None

        self.lstm_cells = []
        for k in xrange(max_input_sequence_len):
            cell = LstmCell(self.W['z'], self.R['z'],
                            self.W['i'], self.R['i'], self.p['i'],
                            self.W['f'], self.R['f'], self.p['f'],
                            self.W['o'], self.R['o'], self.p['o'],
                            self.c[:, k], self.h[:, k],
                            self.dL_dpre['z'][:, k],
                            self.dL_dpre['i'][:, k],
                            self.dL_dpre['f'][:, k],
                            self.dL_dpre['o'][:, k],
                            self.context['z'],
                            self.context['i'],
                            self.context['f'],
                            self.context['o'])
            if k == 0:
                zero_vector = Connector(Matrix.from_npa(np.zeros((p_init.nrows, 1))))
                cell.register_inputs(zero_vector, zero_vector, propagate_error=False)
            else:
                prev_cell = self.lstm_cells[-1]
                cell.register_inputs(prev_cell.c, prev_cell.h)
            self.lstm_cells.append(cell)

    def set_training_mode(self):
        for cell in self.lstm_cells:
            cell.back_prop = True

    def set_testing_mode(self):
        for cell in self.lstm_cells:
            cell.back_prop = False

    def synchronize(self):
        for context in self.context.itervalues():
            context.synchronize()

    def fprop(self, input_sequence):
        n = input_sequence.ncols
        if n > self.max_input_sequence_len:
            raise ValueError('Sequence has length: {} that is too long. '
                             'The maximum is: {}'.
                             format(n, self.max_input_sequence_len))
        for node in 'zifo':
            pre = self.pre[node][:, :n]
            pre.assign_dot(self.context[node], self.W[node], input_sequence)

        for t in xrange(n):
            pre_z = self.pre_columns['z'][t]
            pre_i = self.pre_columns['i'][t]
            pre_f = self.pre_columns['f'][t]
            pre_o = self.pre_columns['o'][t]
            self.lstm_cells[t].fprop(pre_z, pre_i, pre_f, pre_o)

    def bprop(self, input_sequence):
        pass

    def register_inputs(self, x, propagate_error=True):
        self.x = x
        if propagate_error:
            x.register_user(self, context)
        self.propagate_error = propagate_error