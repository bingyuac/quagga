import numpy as np
from network import LstmBlock
from network import MatrixClass, MatrixContextClass


if __name__ == '__main__':
    pt = 'gpu'
    LstmBlock.p_type = pt
    n = 1024
    rng = np.random.RandomState(seed=42)
    a = (4 * rng.rand(n, n) - 2).astype(dtype=np.float32)
    b = (4 * rng.rand(n, 1) - 2).astype(dtype=np.float32)
    c = (4 * rng.rand(n, 1) - 2).astype(dtype=np.float32)

    a = np.asfortranarray(a, dtype=np.float32)
    b = np.asfortranarray(b, dtype=np.float32)
    c = np.asfortranarray(c, dtype=np.float32)

    z_context = MatrixContextClass[pt]()
    i_context = MatrixContextClass[pt]()
    f_context = MatrixContextClass[pt]()
    c_context = MatrixContextClass[pt]()
    o_context = MatrixContextClass[pt]()

    Wz = MatrixClass[pt].from_npa(a)
    Rz = MatrixClass[pt].from_npa(a)
    Wi = MatrixClass[pt].from_npa(a)
    Ri = MatrixClass[pt].from_npa(a)
    pi = MatrixClass[pt].from_npa(b)
    Wf = MatrixClass[pt].from_npa(a)
    Rf = MatrixClass[pt].from_npa(a)
    pf = MatrixClass[pt].from_npa(b)
    Wo = MatrixClass[pt].from_npa(a)
    Ro = MatrixClass[pt].from_npa(a)
    po = MatrixClass[pt].from_npa(b)

    c_t = MatrixClass[pt].from_npa(b)
    h_t = MatrixClass[pt].from_npa(b)
    dL_dpre_z_t = MatrixClass[pt].from_npa(b)
    dL_dpre_i_t = MatrixClass[pt].from_npa(b)
    dL_dpre_f_t = MatrixClass[pt].from_npa(b)
    dL_dpre_o_t = MatrixClass[pt].from_npa(b)

    lstm_block = LstmBlock(Wz, Rz, Wi, Ri, pi, Wf, Rf, pf, Wo, Ro, po, c_t, h_t,
                           dL_dpre_z_t, dL_dpre_i_t, dL_dpre_f_t, dL_dpre_o_t,
                           z_context, i_context, f_context, c_context, o_context)

    pre_z_t = MatrixClass[pt].from_npa(c)
    pre_i_t = MatrixClass[pt].from_npa(c)
    pre_f_t = MatrixClass[pt].from_npa(c)
    pre_o_t = MatrixClass[pt].from_npa(c)
    h_tm1 = MatrixClass[pt].from_npa(c)
    c_tm1 = MatrixClass[pt].from_npa(c)

    lstm_block.back_prop = True
    #
    for i in xrange(50):
        lstm_block.forward_propagation(pre_z_t, pre_i_t, pre_f_t, pre_o_t, h_tm1, c_tm1)