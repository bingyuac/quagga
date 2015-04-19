#include "BidirectionalLstmRnn.h"


BidirectionalLstmRnn::BidirectionalLstmRnn(MatrixGenerator *W_init,
										   MatrixGenerator *R_init,
										   MatrixGenerator *p_init,
										   MatrixGenerator *logistic_init) :
	Wz_f(W_init->get_matrix()), Wz_b(W_init->get_matrix()),
	Rz_f(R_init->get_matrix()), Rz_b(R_init->get_matrix()),
	Wi_f(W_init->get_matrix()), Wi_b(W_init->get_matrix()),
	Ri_f(R_init->get_matrix()), Ri_b(R_init->get_matrix()),
	pi_f(p_init->get_matrix()), pi_b(p_init->get_matrix()),
	Wf_f(W_init->get_matrix()), Wf_b(W_init->get_matrix()),
	Rf_f(R_init->get_matrix()), Rf_b(R_init->get_matrix()),
	pf_f(p_init->get_matrix()), pf_b(p_init->get_matrix()),
	Wo_f(W_init->get_matrix()), Wo_b(W_init->get_matrix()),
	Ro_f(R_init->get_matrix()), Ro_b(R_init->get_matrix()),
	po_f(p_init->get_matrix()), po_b(p_init->get_matrix()),
	w_hy_f(logistic_init->get_matrix()), w_hy_b(logistic_init->get_matrix()),

	pre_z_f(new GpuMatrix(p_init->nrows, kMaxSentenceLen)), pre_z_b(new GpuMatrix(p_init->nrows, kMaxSentenceLen)),
	pre_i_f(new GpuMatrix(p_init->nrows, kMaxSentenceLen)), pre_i_b(new GpuMatrix(p_init->nrows, kMaxSentenceLen)),
	pre_f_f(new GpuMatrix(p_init->nrows, kMaxSentenceLen)), pre_f_b(new GpuMatrix(p_init->nrows, kMaxSentenceLen)),
	c_f(new GpuMatrix(p_init->nrows, kMaxSentenceLen)), c_b(new GpuMatrix(p_init->nrows, kMaxSentenceLen)),
	pre_o_f(new GpuMatrix(p_init->nrows, kMaxSentenceLen)), pre_o_b(new GpuMatrix(p_init->nrows, kMaxSentenceLen)),
	h_f(new GpuMatrix(p_init->nrows, kMaxSentenceLen)), h_b(new GpuMatrix(p_init->nrows, kMaxSentenceLen)),
	pre_z_f_columns(pre_z_f->getColumns()), pre_z_b_columns(pre_z_b->getColumns()),
	pre_i_f_columns(pre_i_f->getColumns()), pre_i_b_columns(pre_i_b->getColumns()),
	pre_f_f_columns(pre_f_f->getColumns()), pre_f_b_columns(pre_f_b->getColumns()),
	c_f_columns(c_f->getColumns()), c_b_columns(c_b->getColumns()),
	pre_o_f_columns(pre_o_f->getColumns()), pre_o_b_columns(pre_o_b->getColumns()),
	h_f_columns(h_f->getColumns()), h_b_columns(h_b->getColumns()),

	dL_dh_f(new GpuMatrix(p_init->nrows, p_init->ncols)), dL_dh_b(new GpuMatrix(p_init->nrows, p_init->ncols)),

	dL_dpre_z_f(new GpuMatrix(p_init->nrows, kMaxSentenceLen)), dL_dpre_z_b(new GpuMatrix(p_init->nrows, kMaxSentenceLen)),
	dL_dpre_i_f(new GpuMatrix(p_init->nrows, kMaxSentenceLen)), dL_dpre_i_b(new GpuMatrix(p_init->nrows, kMaxSentenceLen)),
	dL_dpre_f_f(new GpuMatrix(p_init->nrows, kMaxSentenceLen)), dL_dpre_f_b(new GpuMatrix(p_init->nrows, kMaxSentenceLen)),
	dL_dpre_o_f(new GpuMatrix(p_init->nrows, kMaxSentenceLen)), dL_dpre_o_b(new GpuMatrix(p_init->nrows, kMaxSentenceLen)),
	dL_dpre_z_f_columns(dL_dpre_z_f->getColumns()), dL_dpre_z_b_columns(dL_dpre_z_b->getColumns()),
	dL_dpre_i_f_columns(dL_dpre_i_f->getColumns()), dL_dpre_i_b_columns(dL_dpre_i_b->getColumns()),
	dL_dpre_f_f_columns(dL_dpre_f_f->getColumns()), dL_dpre_f_b_columns(dL_dpre_f_b->getColumns()),
	dL_dpre_o_f_columns(dL_dpre_o_f->getColumns()), dL_dpre_o_b_columns(dL_dpre_o_b->getColumns()),

	dL_dWz_f(new GpuMatrix(W_init->nrows, W_init->ncols)), dL_dWz_b(new GpuMatrix(W_init->nrows, W_init->ncols)),
	dL_dRz_f(new GpuMatrix(R_init->nrows, R_init->ncols)), dL_dRz_b(new GpuMatrix(R_init->nrows, R_init->ncols)),
	dL_dWi_f(new GpuMatrix(W_init->nrows, W_init->ncols)), dL_dWi_b(new GpuMatrix(W_init->nrows, W_init->ncols)),
	dL_dRi_f(new GpuMatrix(R_init->nrows, R_init->ncols)), dL_dRi_b(new GpuMatrix(R_init->nrows, R_init->ncols)),
	dL_dpi_f(new GpuMatrix(p_init->nrows, p_init->ncols)), dL_dpi_b(new GpuMatrix(p_init->nrows, p_init->ncols)),
	dL_dWf_f(new GpuMatrix(W_init->nrows, W_init->ncols)), dL_dWf_b(new GpuMatrix(W_init->nrows, W_init->ncols)),
	dL_dRf_f(new GpuMatrix(R_init->nrows, R_init->ncols)), dL_dRf_b(new GpuMatrix(R_init->nrows, R_init->ncols)),
	dL_dpf_f(new GpuMatrix(p_init->nrows, p_init->ncols)), dL_dpf_b(new GpuMatrix(p_init->nrows, p_init->ncols)),
	dL_dWo_f(new GpuMatrix(W_init->nrows, W_init->ncols)), dL_dWo_b(new GpuMatrix(W_init->nrows, W_init->ncols)),
	dL_dRo_f(new GpuMatrix(R_init->nrows, R_init->ncols)), dL_dRo_b(new GpuMatrix(R_init->nrows, R_init->ncols)),
	dL_dpo_f(new GpuMatrix(p_init->nrows, p_init->ncols)), dL_dpo_b(new GpuMatrix(p_init->nrows, p_init->ncols)),
	dL_dw_hy_f(new GpuMatrix(logistic_init->nrows, logistic_init->ncols)), dL_dw_hy_b(new GpuMatrix(logistic_init->nrows, logistic_init->ncols)),
	dL_dx_f(new GpuMatrix(p_init->nrows, kMaxSentenceLen)), dL_dx(new GpuMatrix(p_init->nrows, kMaxSentenceLen)) {

	z_f_context = new GpuMatrixContext(); z_b_context = new GpuMatrixContext();
	i_f_context = new GpuMatrixContext(); i_b_context = new GpuMatrixContext();
	f_f_context = new GpuMatrixContext(); f_b_context = new GpuMatrixContext();
	c_f_context = new GpuMatrixContext(); c_b_context = new GpuMatrixContext();
	o_f_context = new GpuMatrixContext(); o_b_context = new GpuMatrixContext();

	for (int n = 0; n < kMaxSentenceLen; n++) {
		forward_lstm_block[n] = new LstmBlock(Wz_f, Rz_f, Wi_f, Ri_f, pi_f, Wf_f, Rf_f, pf_f, Wo_f, Ro_f, po_f,
											  c_f_columns[n], h_f_columns[n],
											  dL_dpre_z_f_columns[n],
											  dL_dpre_i_f_columns[n],
											  dL_dpre_f_f_columns[n],
											  dL_dpre_o_f_columns[n],
											  z_f_context,
											  i_f_context,
											  f_f_context,
											  c_f_context,
											  o_f_context);
		backward_lstm_block[n] = new LstmBlock(Wz_b, Rz_b, Wi_b, Ri_b, pi_b, Wf_b, Rf_b, pf_b, Wo_b, Ro_b, po_b,
											   c_b_columns[n], h_b_columns[n],
											   dL_dpre_z_b_columns[n],
											   dL_dpre_i_b_columns[n],
											   dL_dpre_f_b_columns[n],
											   dL_dpre_o_b_columns[n],
											   z_b_context,
											   i_b_context,
											   f_b_context,
											   c_b_context,
											   o_b_context);
	}

	zero_vector = new GpuMatrix(p_init->nrows, 1);
	zero_vector->scale(0.0f, z_f_context);
	z_f_context->synchronize();
}


BidirectionalLstmRnn::~BidirectionalLstmRnn(void) {
	for (int n = 0; n < kMaxSentenceLen; n++) {
		delete forward_lstm_block[n];
		delete backward_lstm_block[n];
	}
	delete zero_vector;

	delete Wz_f; delete Wz_b;
	delete Rz_f; delete Rz_b;
	delete Wi_f; delete Wi_b;
	delete Ri_f; delete Ri_b;
	delete pi_f; delete pi_b;
	delete Wf_f; delete Wf_b;
	delete Rf_f; delete Rf_b;
	delete pf_f; delete pf_b;
	delete Wo_f; delete Wo_b;
	delete Ro_f; delete Ro_b;
	delete po_f; delete po_b;
	delete w_hy_f; delete w_hy_b;

	delete pre_z_f; delete pre_z_b;
	delete pre_i_f; delete pre_i_b;
	delete pre_f_f; delete pre_f_b;
	delete c_f; delete c_b;
	delete pre_o_f; delete pre_o_b;
	delete h_f, delete h_b;
	for (int i = 0; i < kMaxSentenceLen; i++) {
		delete pre_z_f_columns[i]; delete pre_z_b_columns[i];
		delete pre_i_f_columns[i]; delete pre_i_b_columns[i];
		delete pre_f_f_columns[i]; delete pre_f_b_columns[i];
		delete c_f_columns[i]; delete c_b_columns[i];
		delete pre_o_f_columns[i]; delete pre_o_b_columns[i];
		delete h_f_columns[i]; delete h_b_columns[i];
	}
	delete []pre_z_f_columns; delete []pre_z_b_columns;
	delete []pre_i_f_columns; delete []pre_i_b_columns;
	delete []pre_f_f_columns; delete []pre_f_b_columns;
	delete []c_f_columns; delete []c_b_columns;
	delete []pre_o_f_columns; delete []pre_o_b_columns;
	delete []h_f_columns; delete []h_b_columns;

	delete dL_dh_f; delete dL_dh_b;

	delete dL_dpre_z_f; delete dL_dpre_z_b;
	delete dL_dpre_i_f; delete dL_dpre_i_b;
	delete dL_dpre_f_f; delete dL_dpre_f_b;
	delete dL_dpre_o_f; delete dL_dpre_o_b;
	for (int i = 0; i < kMaxSentenceLen; i++) {
		delete dL_dpre_z_f_columns[i]; delete dL_dpre_z_b_columns[i];
		delete dL_dpre_i_f_columns[i]; delete dL_dpre_i_b_columns[i];
		delete dL_dpre_f_f_columns[i]; delete dL_dpre_f_b_columns[i];
		delete dL_dpre_o_f_columns[i]; delete dL_dpre_o_b_columns[i];
	}
	delete []dL_dpre_z_f_columns; delete []dL_dpre_z_b_columns;
	delete []dL_dpre_i_f_columns; delete []dL_dpre_i_b_columns;
	delete []dL_dpre_f_f_columns; delete []dL_dpre_f_b_columns;
	delete []dL_dpre_o_f_columns; delete []dL_dpre_o_b_columns;

	delete dL_dWz_f; delete dL_dWz_b;
	delete dL_dRz_f; delete dL_dRz_b;
	delete dL_dWi_f; delete dL_dWi_b;
	delete dL_dRi_f; delete dL_dRi_b;
	delete dL_dpi_f; delete dL_dpi_b;
	delete dL_dWf_f; delete dL_dWf_b;
	delete dL_dRf_f; delete dL_dRf_b;
	delete dL_dpf_f; delete dL_dpf_b;
	delete dL_dWo_f; delete dL_dWo_b;
	delete dL_dRo_f; delete dL_dRo_b;
	delete dL_dpo_f; delete dL_dpo_b;
	delete dL_dw_hy_f; delete dL_dw_hy_b;
	delete dL_dx_f; delete dL_dx;

	GpuMatrixContext::destroyCublasHandle();
	delete z_f_context; delete z_b_context;
	delete i_f_context; delete i_b_context;
	delete f_f_context; delete f_b_context;
	delete c_f_context; delete c_b_context;
	delete o_f_context; delete o_b_context;
}



float BidirectionalLstmRnn::forwardPropagation(GpuMatrix *sentence) {
	int n = sentence->ncols;
	int input_dim = pi_f->nrows;

	GpuMatrix subm_pre_z_f(pre_z_f->data, input_dim, n);
	GpuMatrix subm_pre_z_b(pre_z_b->data, input_dim, n);
	GpuMatrix subm_pre_i_f(pre_i_f->data, input_dim, n);
	GpuMatrix subm_pre_i_b(pre_i_b->data, input_dim, n);
	GpuMatrix subm_pre_f_f(pre_f_f->data, input_dim, n);
	GpuMatrix subm_pre_f_b(pre_f_b->data, input_dim, n);
	GpuMatrix subm_pre_o_f(pre_o_f->data, input_dim, n);
	GpuMatrix subm_pre_o_b(pre_o_b->data, input_dim, n);
	Wz_f->dot(sentence, &subm_pre_z_f, z_f_context);
	Wz_b->dot(sentence, &subm_pre_z_b, z_b_context);
	Wi_f->dot(sentence, &subm_pre_i_f, i_f_context);
	Wi_b->dot(sentence, &subm_pre_i_b, i_b_context);
	Wf_f->dot(sentence, &subm_pre_f_f, f_f_context);
	Wf_b->dot(sentence, &subm_pre_f_b, f_b_context);
	Wo_f->dot(sentence, &subm_pre_o_f, o_f_context);
	Wo_b->dot(sentence, &subm_pre_o_b, o_b_context);

	forward_lstm_block[0]->forwardPropagation(pre_z_f_columns[0], pre_i_f_columns[0], pre_f_f_columns[0], pre_o_f_columns[0], zero_vector, zero_vector);
	backward_lstm_block[0]->forwardPropagation(pre_z_b_columns[0], pre_i_b_columns[0], pre_f_b_columns[0], pre_o_b_columns[0], zero_vector, zero_vector);
	for (int t = 1; t < n; t++) {
		forward_lstm_block[t]->forwardPropagation(pre_z_f_columns[t], pre_i_f_columns[t], pre_f_f_columns[t], pre_o_f_columns[t], forward_lstm_block[t-1]->h_t, forward_lstm_block[t-1]->c_t);
		backward_lstm_block[t]->forwardPropagation(pre_z_f_columns[n-1-t], pre_i_f_columns[n-1-t], pre_f_b_columns[t], pre_o_f_columns[n-1-t], backward_lstm_block[t-1]->h_t, backward_lstm_block[t-1]->c_t);
	}

	float pre_exp = forward_lstm_block[n-1]->h_t->vdot(w_hy_f, o_f_context) +
			        backward_lstm_block[n-1]->h_t->vdot(w_hy_b, o_b_context);

	return 1.0f / (1.0f + exp(-pre_exp));
}


void BidirectionalLstmRnn::backwardPropagation(GpuMatrix *sentence, int sentence_grammaticality) {
	float predicted_sentence_grammaticality = forwardPropagation(sentence);
	float error = sentence_grammaticality - predicted_sentence_grammaticality;

	GpuMatrix *c_f_tm1;
	GpuMatrix *c_b_tm1;
	int n = sentence->ncols;
	for (int t = 0; t < n; t++) {
		if (t == n - 1) {
			c_f_tm1 = zero_vector;
			c_b_tm1 = zero_vector;
		} else {
			c_f_tm1 = forward_lstm_block[n-2-t]->c_t;
			c_b_tm1 = backward_lstm_block[n-2-t]->c_t;
		}

		if (t == 0) {
			forward_lstm_block[n-1]->h_t->scale(error, dL_dh_f, o_f_context);
			backward_lstm_block[n-1]->h_t->scale(error, dL_dh_b, o_b_context);
			forward_lstm_block[n-1]->backwardPropagation(dL_dh_f, c_f_tm1);
			backward_lstm_block[n-1]->backwardPropagation(dL_dh_b, c_b_tm1);
		} else {
			forward_lstm_block[n-1-t]->backwardPropagation(forward_lstm_block[n-t]->dL_dpre_z_t,
														   forward_lstm_block[n-t]->dL_dpre_i_t,
														   forward_lstm_block[n-t]->dL_dpre_f_t,
														   forward_lstm_block[n-t]->dL_dpre_o_t,
														   forward_lstm_block[n-t]->dL_dc_t,
														   forward_lstm_block[n-t]->f_t,
														   c_f_tm1);
			backward_lstm_block[n-1-t]->backwardPropagation(backward_lstm_block[n-t]->dL_dpre_z_t,
															backward_lstm_block[n-t]->dL_dpre_i_t,
															backward_lstm_block[n-t]->dL_dpre_f_t,
															backward_lstm_block[n-t]->dL_dpre_o_t,
															backward_lstm_block[n-t]->dL_dc_t,
															backward_lstm_block[n-t]->f_t,
															c_b_tm1);
		}
	}

	// dL/dx = Wz_f.T * dL/dpre_z_f + Wi_f.T * dL/dpre_i_f + Wf_f.T * dL/dpre_f_f + Wo_f.T * dL/dpre_o_f +
	//		   Wz_b.T * dL/dpre_z_b + Wi_b.T * dL/dpre_i_b + Wf_b.T * dL/dpre_f_b + Wo_b.T * dL/dpre_o_b

	int input_dim = pi_f->nrows;
	GpuMatrix subm_dL_dpre_z_f(dL_dpre_z_f->data, input_dim, n);
	GpuMatrix subm_dL_dpre_i_f(dL_dpre_i_f->data, input_dim, n);
	GpuMatrix subm_dL_dpre_f_f(dL_dpre_f_f->data, input_dim, n);
	GpuMatrix subm_dL_dpre_o_f(dL_dpre_o_f->data, input_dim, n);
	GpuMatrix subm_dL_dpre_z_b(dL_dpre_z_b->data, input_dim, n);
	GpuMatrix subm_dL_dpre_i_b(dL_dpre_i_b->data, input_dim, n);
	GpuMatrix subm_dL_dpre_f_b(dL_dpre_f_b->data, input_dim, n);
	GpuMatrix subm_dL_dpre_o_b(dL_dpre_o_b->data, input_dim, n);

	Wz_f->tdot(&subm_dL_dpre_z_f, dL_dx_f, z_f_context);
	Wi_f->tdot(&subm_dL_dpre_i_f, 1.0f, dL_dx_f, z_f_context);
	Wf_f->tdot(&subm_dL_dpre_f_f, 1.0f, dL_dx_f, z_f_context);
	Wo_f->tdot(&subm_dL_dpre_o_f, 1.0f, dL_dx_f, z_f_context);

	Wz_b->tdot(&subm_dL_dpre_z_b, dL_dx, z_b_context);
	Wi_b->tdot(&subm_dL_dpre_i_b, 1.0f, dL_dx, z_b_context);
	Wf_b->tdot(&subm_dL_dpre_f_b, 1.0f, dL_dx, z_b_context);
	Wo_b->tdot(&subm_dL_dpre_o_b, 1.0f, dL_dx, z_b_context);

	// dL_dW@ = dL_dpre_@ * x;
	subm_dL_dpre_z_f.dot(sentence, dL_dWz_f, z_f_context);
	subm_dL_dpre_i_f.dot(sentence, dL_dWi_f, i_f_context);
	subm_dL_dpre_f_f.dot(sentence, dL_dWf_f, f_f_context);
	subm_dL_dpre_o_f.dot(sentence, dL_dWo_f, o_f_context);
	subm_dL_dpre_z_b.dot(sentence, dL_dWz_b, z_b_context);
	subm_dL_dpre_i_b.dot(sentence, dL_dWi_b, i_b_context);
	subm_dL_dpre_f_b.dot(sentence, dL_dWf_b, f_b_context);
	subm_dL_dpre_o_b.dot(sentence, dL_dWo_b, o_b_context);

	// dL_dR@ = dL_dpre_@ * h;
	GpuMatrix subm_h_f(h_f->data, input_dim, n);
	GpuMatrix subm_h_b(h_b->data, input_dim, n);

	subm_dL_dpre_z_f.dot(&subm_h_f, dL_dRz_f, z_f_context);
	subm_dL_dpre_i_f.dot(&subm_h_f, dL_dRi_f, i_f_context);
	subm_dL_dpre_f_f.dot(&subm_h_f, dL_dRf_f, f_f_context);
	subm_dL_dpre_o_f.dot(&subm_h_f, dL_dRo_f, o_f_context);
	subm_dL_dpre_z_b.dot(&subm_h_b, dL_dRz_b, z_b_context);
	subm_dL_dpre_i_b.dot(&subm_h_b, dL_dRi_b, i_b_context);
	subm_dL_dpre_f_b.dot(&subm_h_b, dL_dRf_b, f_b_context);
	subm_dL_dpre_o_b.dot(&subm_h_b, dL_dRo_b, o_b_context);

	// dL_dp@ = c * dL_dpre_@;
	GpuMatrix subm_c_f(c_f->data, input_dim, n);
	GpuMatrix subm_c_b(c_b->data, input_dim, n);

	subm_c_f.dot(&subm_dL_dpre_i_f, dL_dpi_f, i_f_context);
	subm_c_f.dot(&subm_dL_dpre_f_f, dL_dpf_f, f_f_context);
	subm_c_f.dot(&subm_dL_dpre_o_f, dL_dpo_f, o_f_context);
	subm_c_b.dot(&subm_dL_dpre_i_b, dL_dpi_b, i_b_context);
	subm_c_b.dot(&subm_dL_dpre_f_b, dL_dpf_b, f_b_context);
	subm_c_b.dot(&subm_dL_dpre_o_b, dL_dpo_b, o_b_context);

	z_f_context->synchronize();
	i_f_context->synchronize();
	f_f_context->synchronize();
	o_f_context->synchronize();
	z_b_context->synchronize();
	i_b_context->synchronize();
	f_b_context->synchronize();
	dL_dx->add(dL_dx_f, o_b_context);
	o_b_context->synchronize();
}
