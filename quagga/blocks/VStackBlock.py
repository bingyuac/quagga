from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import Connector


class VStackBlock(object):
    def __init__(self, f_matrix, s_matrix, max_ncols):
        if f_matrix.dtype != s_matrix.dtype:
            raise ValueError("Can't stack matrices with different dtypes!")

        self.f_matrix = f_matrix
        self.s_matrix = s_matrix
        self.max_ncols = max_ncols

        nrows = f_matrix.nrows + s_matrix.nrows
        dtype = f_matrix.dtype

        self.buffer = Matrix.empty(nrows, max_ncols, dtype)
        self.context = Context()
        self.output = Connector(self.buffer, self.context)

        self.dL_df_buffer = Matrix.empty(f_matrix.nrows, max_ncols, dtype)
        self.dL_ds_buffer = Matrix.empty(s_matrix.nrows, max_ncols, dtype)
        f_matrix.register_user(self, self.context, self.dL_df_buffer)
        s_matrix.register_user(self, self.context, self.dL_ds_buffer)

    def fprop(self):
        if self.f_matrix.ncols > self.max_ncols or \
                self.s_matrix.ncols > self.max_ncols:
            raise ValueError('One of the matrix is too big!')
        output = self.buffer[:, self.f_matrix.ncols]
        self.f_matrix.block(self.context)
        self.s_matrix.block(self.context)
        output.assign_vstack(self.context, self.f_matrix, self.s_matrix)
        self.output.matrix = output

    def bprop(self):
        self.output.backward_block(self.context)
        dL_doutput = self.output.derivative