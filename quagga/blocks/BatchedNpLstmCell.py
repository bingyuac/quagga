import numpy as np
from quagga.matrix import Matrix
from quagga.context import Context
from quagga.blocks import Connector


class BatchedNpLstmCell(object):
    def __init__(self, W, R, h, pre_zifo, dL_dpre_zifo, prev_c, prev_h, context, propagate_error=True):
        pass