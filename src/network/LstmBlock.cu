#include "LstmBlock.h"


LstmBlock::LstmBlock(GpuMatrix *Wz, GpuMatrix *Rz,
					 GpuMatrix *Wi, GpuMatrix *Ri, GpuMatrix *pi,
					 GpuMatrix *Wf, GpuMatrix *Rf, GpuMatrix *pf,
					 GpuMatrix *Wo, GpuMatrix *Ro, GpuMatrix *po,
					 GpuMatrix *dL_dpre_z_t, GpuMatrix *dL_dpre_i_t, GpuMatrix *dL_dpre_f_t, GpuMatrix *dL_dpre_o_t,
					 GpuMatrixContext *z_context, GpuMatrixContext *i_context, GpuMatrixContext *f_context, GpuMatrixContext *c_context, GpuMatrixContext *o_context) :
	 Wz(Wz),
	 Rz(Rz),
	 Wi(Wi),
	 Ri(Ri),
	 pi(pi),
	 Wf(Wf),
	 Rf(Rf),
	 pf(pf),
	 Wo(Wo),
	 Ro(Ro),
	 po(po),
	 z_t(new GpuMatrix(pi->nrows, 1)),
	 i_t(new GpuMatrix(pi->nrows, 1)),
	 f_t(new GpuMatrix(pi->nrows, 1)),
	 c_t(new GpuMatrix(pi->nrows, 1)),
	 tanh_c_t(new GpuMatrix(pi->nrows, 1)),
	 o_t(new GpuMatrix(pi->nrows, 1)),
	 h_t(new GpuMatrix(pi->nrows, 1)),
	 dz_t_dpre_z_t(new GpuMatrix(pi->nrows, 1)),
	 di_t_dpre_i_t(new GpuMatrix(pi->nrows, 1)),
	 df_t_dpre_f_t(new GpuMatrix(pi->nrows, 1)),
	 dtanh_c_t_dc_t(new GpuMatrix(pi->nrows, 1)),
	 do_t_dpre_o_t(new GpuMatrix(pi->nrows, 1)),
	 dL_dh_t(new GpuMatrix(pi->nrows, 1)),
	 dL_dhz_t(new GpuMatrix(pi->nrows, 1)),
	 dL_dhi_t(new GpuMatrix(pi->nrows, 1)),
	 dL_dhf_t(new GpuMatrix(pi->nrows, 1)),
	 dL_dc_t(new GpuMatrix(pi->nrows, 1)),
	 dL_dpre_o_t(dL_dpre_o_t),
	 dL_dpre_f_t(dL_dpre_f_t),
	 dL_dpre_i_t(dL_dpre_i_t),
	 dL_dpre_z_t(dL_dpre_z_t),
	 z_context(z_context),
	 i_context(i_context),
	 f_context(f_context),
	 c_context(c_context),
	 o_context(o_context) {}


LstmBlock::~LstmBlock(void) {
	delete z_t;
	delete i_t;
	delete f_t;
	delete c_t;
	delete tanh_c_t;
	delete o_t;
	delete h_t;
	delete dtanh_c_t_dc_t;
	delete dz_t_dpre_z_t;
	delete di_t_dpre_i_t;
	delete df_t_dpre_f_t;
	delete do_t_dpre_o_t;
	delete dL_dh_t;
	delete dL_dhz_t;
	delete dL_dhi_t;
	delete dL_dhf_t;
	delete dL_dc_t;
}


void LstmBlock::forwardPropagation(GpuMatrix *pre_z_t, GpuMatrix *pre_i_t, GpuMatrix *pre_f_t, GpuMatrix *pre_o_t, GpuMatrix *h_tm1, GpuMatrix *c_tm1, bool back_prop) {
	// z[t] = tanh(Wz * x[t] + Rz * h[t-1])
	// pre_z[t] = Wz * x[t]
	// pre_z[t] += Rz * h[t-1]
	// z[t] = tanh(pre_z[t])
	Rz->dot(h_tm1, 1.0f, pre_z_t, z_context);
	if (back_prop) {
		pre_z_t->tanh(z_t, dz_t_dpre_z_t, z_context);
	} else {
		pre_z_t->tanh(z_t, z_context);
	}

	// i[t] = sigmoid(Wi * x[t] + Ri * h[t-1] + pi .* c[t-1])
	// pre_i[t] = Wi * x[t]
	// pre_i[t] += Ri * h[t-1]
	// pre_i[t] += pi .* c[t-1]
	// i[t] = sigmoid(pre_i[t])
	Ri->dot(h_t, 1.0f, pre_i_t, i_context);
	pi->hdot(c_tm1, 1.0f, pre_i_t, i_context);
	if (back_prop) {
		pre_i_t->sigmoid(i_t, di_t_dpre_i_t, i_context);
	} else {
		pre_i_t->sigmoid(i_t, i_context);
	}

	// f[t] = sigmoid(Wf * x[t] + Rf * h[t-1] + pf .* c[t-1])
	// pre_f[t] = Wf * x[t]
	// pre_f[t] += Rf * h[t-1]
	// pre_f[t] += pf .* c[t-1]
	// f[t] = sigmoid(pre_f[t])
	Rf->dot(h_tm1, 1.0f, pre_f_t, f_context);
	pf->hdot(c_tm1, 1.0f, pre_f_t, f_context);
	if (back_prop) {
		pre_f_t->sigmoid(f_t, df_t_dpre_f_t, f_context);
	} else {
		pre_f_t->sigmoid(f_t, f_context);
	}

	// c[t] = i[t] .* z[t] + f[t] .* c[t-1]
	z_context->synchronize();
	i_context->synchronize();
	f_context->synchronize();
	GpuMatrix::addHdot(i_t, z_t, f_t, c_tm1, c_t, c_context);

	// o[t] = sigmoid(Wo * x[t] + Ro * h[t-1] + po .* c[t])
	// pre_o[t] = Wo * x[t]
	// pre_o[t] += Ro * h[t-1]
	// pre_o[t] += po .* c[t]
	// o[t] = sigmoid(pre_o[t])
	Ro->dot(h_tm1, 1.0f, pre_o_t, o_context);
	c_context->synchronize();
	po->hdot(c_t, 1.0f, pre_o_t, o_context);
	if (back_prop) {
		pre_o_t->sigmoid(o_t, do_t_dpre_o_t, o_context);
	} else {
		pre_o_t->sigmoid(o_t, o_context);
	}

	// h[t] = o[t] .* tanh(c[t])
	// tanh_c[t] = tanh(c[t])
	// h[t] = o[t] .* tanh_c_t
	if (back_prop) {
		c_t->tanh(tanh_c_t, dtanh_c_t_dc_t, c_context);
	} else {
		c_t->tanh(tanh_c_t, c_context);
	}
	c_context->synchronize();
	tanh_c_t->hdot(o_t, 0.0f, h_t, o_context);
	o_context->synchronize();
}


void LstmBlock::backwardPropagation(GpuMatrix *dL_dh_t, GpuMatrix *c_tm1) {
	// dL/dpre_o[t] = dL/dh[t] .* tanh(c[t]) .* do[t]/dpre_o[t]
	GpuMatrix::hdot(dL_dh_t, tanh_c_t, do_t_dpre_o_t, dL_dpre_o_t, o_context);

	// dL/dc[t] = dL/dh[t] .* o[t] .* dtanh(c[t])/dc[t] + po .* dL/dpre_o[t]
	GpuMatrix::addHdot(dL_dh_t, o_t, dtanh_c_t_dc_t, po, dL_dpre_o_t, dL_dc_t, o_context);

	// dL/dpre_f[t] = dL/dc[t] .* c[t-1] .* df[t]/dpre_f[t]
	// dL/dpre_i[t] = dL/dc[t] .* z[t] .* di[t]/dpre_i[t]
	// dL/dpre_z[t] = dL/dc[t] .* i[t] .* dz[t]/dpre_z[t]
	o_context->synchronize();
	GpuMatrix::hdot(dL_dc_t, c_tm1, df_t_dpre_f_t, dL_dpre_f_t, f_context);
	GpuMatrix::hdot(dL_dc_t, z_t, di_t_dpre_i_t, dL_dpre_i_t, i_context);
	GpuMatrix::hdot(dL_dc_t, i_t, dz_t_dpre_z_t, dL_dpre_z_t, z_context);
}

void LstmBlock::backwardPropagation(GpuMatrix *dL_dpre_z_tp1, GpuMatrix *dL_dpre_i_tp1, GpuMatrix *dL_dpre_f_tp1, GpuMatrix *dL_dpre_o_tp1, GpuMatrix *dL_dc_tp1, GpuMatrix *f_tp1, GpuMatrix *c_tm1) {
	// Todo add gradient clipping from Alex Graves paper

	//dL/dh[t] = Rz.T * dL/dpre_z[t+1] + Ri.T * dL/dpre_i[t+1] + Rf.T * dL/dpre_f[t+1] + Ro.T * dL/dpre_o[t+1]
	Rz->dot(true, dL_dpre_z_tp1, dL_dhz_t, z_context);
	Ri->dot(true, dL_dpre_i_tp1, dL_dhi_t, i_context);
	Rf->dot(true, dL_dpre_f_tp1, dL_dhf_t, f_context);
	Ro->dot(true, dL_dpre_o_tp1, dL_dh_t, o_context);
	z_context->synchronize();
	i_context->synchronize();
	f_context->synchronize();
	dL_dh_t->add(dL_dhz_t, dL_dhi_t, dL_dhf_t, o_context);


	// dL/dpre_o[t] = dL/dh[t] .* tanh(c[t]) .* do[t]/dpre_o[t]
	GpuMatrix::hdot(dL_dh_t, tanh_c_t, dL_dpre_o_t, dL_dpre_o_t, o_context);

	// dL/dc[t] = dL/dh[t] .* o[t] .* dtanh(c[t])/dc[t] + po .* dL/dpre_o[t] +
	//													  pi .* dL/dpre_i[t+1] +
	//									    			  pf .* dL/dpre_f[t+1] +
	//											    	  dL/dc[t+1] .* f[t+1]
	GpuMatrix::addHdot(dL_dh_t, o_t, dtanh_c_t_dc_t, po, dL_dpre_o_t, pi, dL_dpre_i_tp1, pf, dL_dpre_f_tp1, dL_dc_tp1, f_tp1, o_context);

	// dL/dpre_f[t] = dL/dc[t] .* c[t-1] .* df[t]/dpre_f[t]
	// dL/dpre_i[t] = dL/dc[t] .* z[t] .* di[t]/dpre_i[t]
	// dL/dpre_z[t] = dL/dc[t] .* i[t] .* dz[t]/dpre_z[t]
	o_context->synchronize();
	GpuMatrix::hdot(dL_dc_t, c_tm1, df_t_dpre_f_t, dL_dpre_f_t, f_context);
	GpuMatrix::hdot(dL_dc_t, z_t, di_t_dpre_i_t, dL_dpre_i_t, i_context);
	GpuMatrix::hdot(dL_dc_t, i_t, dz_t_dpre_z_t, dL_dpre_z_t, z_context);
}

