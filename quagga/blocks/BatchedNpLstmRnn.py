import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import Connector


class BatchedNpLstmRnn(object):
    def __init__(self, W_init, R_init, x, propagate_error=True):
        if W_init.nrows != R_init.nrows:
            raise ValueError('W and R have to have the same number of rows!')
        if R_init.nrows != R_init.ncols:
            raise ValueError('R must be a square matrix!')

        nrows = R_init.nrows
        self.context = Context()
        self.max_input_sequence_len = len(x)

        self.W = Matrix.empty(4 * nrows, W_init.ncols, 'float')
        self.W.assign_hstack(self.context, W_init(), W_init(), W_init(), W_init())
        self.dL_dW = Matrix.empty_like(self.W)

        self.R = Matrix.empty(4 * nrows, R_init.ncols, 'float')
        self.R.assign_hstack(self.context, R_init(), R_init(), R_init(), R_init())
        self.dL_dR = Matrix.empty_like(self.R)

        self.x = x
        self.h = []
        self.pre_zifo = []
        self.dL_dpre_zifo = []
        for k, each in enumerate(x):
            each.register_user(self, self.context, self.context)
            # each.get_backward_matrix(self)
            self.h.append(Connector(Matrix.empty(nrows, self.x[k].ncols, 'float'), self.context))
            self.pre_zifo.append(Matrix.empty(4 * nrows, self.x[k].ncols, 'float'))
            self.dL_dpre_zifo.append(Matrix.empty_like(self.pre_zifo))
        self.propagate_error = propagate_error
        self.context.synchronize()

        self.lstm_cells = []
        for k in xrange(len(x)):
            if k == 0:
                prev_c = Matrix.from_npa(np.zeros((nrows, 1)))
                prev_h = prev_c
                propagate_error = False
            else:
                prev_c = self.lstm_cells[-1].c
                prev_h = self.lstm_cells[-1].h
                propagate_error = True
            cell = NpLstmCell(self.R, self.h[k], self.pre_zifo[k], self.dL_dpre_zifo[k], prev_c, prev_h, self.context, propagate_error)
            self.lstm_cells.append(cell)