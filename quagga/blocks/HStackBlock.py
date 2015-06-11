from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import Connector


class HStackBlock(object):
    def __init__(self, f_matrix, s_matrix, f_max_ncols, s_max_ncols):
        if f_matrix.dtype != s_matrix.dtype:
            raise ValueError("Can't stack matrices with different dtypes!")

        self.f_matrix = f_matrix
        self.s_matrix = s_matrix
        self.max_ncols = f_max_ncols + s_max_ncols

        nrows = f_matrix.nrows
        dtype = f_matrix.dtype

        self.buffer = Matrix.empty(nrows, self.max_ncols, dtype)
        self.context = Context()
        self.output = Connector(self.buffer, self.context)

        self.dL_df_buffer = Matrix.empty(nrows, f_max_ncols, dtype)
        self.dL_ds_buffer = Matrix.empty(nrows, s_max_ncols, dtype)
        f_matrix.register_user(self, self.context, self.dL_df_buffer)
        s_matrix.register_user(self, self.context, self.dL_ds_buffer)

    def fprop(self):
        if self.f_matrix.ncols + self.s_matrix.ncols > self.max_ncols:
            raise ValueError('One of the matrix is too big!')
        output = self.buffer[:, self.f_matrix.ncols + self.s_matrix.ncols]
        self.f_matrix.block(self.context)
        self.s_matrix.block(self.context)
        output.assign_hstack(self.context, self.f_matrix, self.s_matrix)
        self.output.matrix = output

    def bprop(self):
        self.dL_df_buffer.ncols = self.f_matrix.ncols
        self.dL_ds_buffer.ncols = self.s_matrix.ncols
        self.output.backward_block(self.context)
        self.output.derivative.hsplit(self.context, self.dL_df_buffer, self.dL_ds_buffer)