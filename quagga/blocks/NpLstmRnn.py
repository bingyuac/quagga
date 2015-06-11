import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import NpLstmCell, Connector


class NpLstmRnn(object):
    def __init__(self, W_init, R_init, max_input_sequence_len):
        if W_init.nrows != R_init.nrows:
            raise ValueError('W and R have to have the same number of rows!')
        if R_init.nrows != R_init.ncols:
            raise ValueError('R must be a square matrix!')
        nrows = R_init.nrows
        self.max_input_sequence_len = max_input_sequence_len
        self.W = Matrix.empty(4 * nrows, W_init.ncols, 'float')
        self.W.assign_hstack(W_init(), W_init(), W_init(), W_init())
        self.dL_dW = Matrix.empty_like(self.W)
        self.R = Matrix.empty(4 * nrows, R_init.ncols, 'float')
        self.R.assign_hstack(R_init(), R_init(), R_init(), R_init())
        self.dL_dR = Matrix.empty_like(self.R)
        self.h = Matrix.empty(nrows, max_input_sequence_len, 'float')
        self.pre_zifo = Matrix.empty_like(self.h)
        self.dL_dpre_zifo = Matrix.empty_like(self.h)
        self.context = Context()

        # self.dL_dx = Matrix.empty(W_init.ncols, max_input_sequence_len)
        self.dL_dx = None

        self.lstm_cells = []
        for k in xrange(max_input_sequence_len):
            if k == 0:
                prev_c = Connector(Matrix.from_npa(np.zeros((nrows, 1))))
                prev_h = prev_c
                propagate_error = False
            else:
                prev_c = self.lstm_cells[-1].c
                prev_h = self.lstm_cells[-1].h
                propagate_error = True
            cell = NpLstmCell(self.W, self.R, self.h[:, k], self.pre_zifo[:, k], self.dL_dpre_zifo[:, k], prev_c, prev_h, self.context, propagate_error)
            self.lstm_cells.append(cell)