#ifndef LSTMBLOCK_H_
#define LSTMBLOCK_H_


#include "../matrix/GpuMatrix.h"
#include "../matrix/GpuMatrixContext.h"


class LstmBlock {
	public:
		LstmBlock(GpuMatrix *Wz, GpuMatrix *Rz,
				  GpuMatrix *Wi, GpuMatrix *Ri, GpuMatrix *pi,
				  GpuMatrix *Wf, GpuMatrix *Rf, GpuMatrix *pf,
				  GpuMatrix *Wo, GpuMatrix *Ro, GpuMatrix *po,
				  GpuMatrix *c_t, GpuMatrix *h_t,
				  GpuMatrix *dL_dpre_z_t, GpuMatrix *dL_dpre_i_t, GpuMatrix *dL_dpre_f_t, GpuMatrix *dL_dpre_o_t,
				  GpuMatrixContext *z_context, GpuMatrixContext *i_context, GpuMatrixContext *f_context, GpuMatrixContext *c_context, GpuMatrixContext *o_context);
		~LstmBlock(void);
		void forwardPropagation(GpuMatrix *pre_z_t,
							    GpuMatrix *pre_i_t,
							    GpuMatrix *pre_f_t,
							    GpuMatrix *pre_o_t,
							    GpuMatrix *h_tm1,
							    GpuMatrix *c_tm1,
							    bool back_prop = true);
		void backwardPropagation(GpuMatrix *dL_dh_t, GpuMatrix *c_tm1);
		void backwardPropagation(GpuMatrix *dL_dpre_z_tp1,
								 GpuMatrix *dL_dpre_i_tp1, GpuMatrix *dL_dpre_f_tp1, GpuMatrix *dL_dpre_o_tp1, GpuMatrix *dL_dc_tp1, GpuMatrix *f_tp1, GpuMatrix *c_tm1);

		GpuMatrix *Wz;
		GpuMatrix *Rz;

		GpuMatrix *Wi;
		GpuMatrix *Ri;
		GpuMatrix *pi;

		GpuMatrix *Wf;
		GpuMatrix *Rf;
		GpuMatrix *pf;

		GpuMatrix *Wo;
		GpuMatrix *Ro;
		GpuMatrix *po;

		GpuMatrix *z_t;
		GpuMatrix *i_t;
		GpuMatrix *f_t;
		GpuMatrix *c_t;
		GpuMatrix *tanh_c_t;
		GpuMatrix *o_t;
		GpuMatrix *h_t;

		GpuMatrix *dz_t_dpre_z_t;
		GpuMatrix *di_t_dpre_i_t;
		GpuMatrix *df_t_dpre_f_t;
		GpuMatrix *dtanh_c_t_dc_t;
		GpuMatrix *do_t_dpre_o_t;

		GpuMatrix *dL_dh_t;
		GpuMatrix *dL_dhz_t;
		GpuMatrix *dL_dhi_t;
		GpuMatrix *dL_dhf_t;
		GpuMatrix *dL_dc_t;

		GpuMatrix *dL_dpre_o_t;
		GpuMatrix *dL_dpre_f_t;
		GpuMatrix *dL_dpre_i_t;
		GpuMatrix *dL_dpre_z_t;

		GpuMatrixContext *z_context;
		GpuMatrixContext *i_context;
		GpuMatrixContext *f_context;
		GpuMatrixContext *c_context;
		GpuMatrixContext *o_context;
};


#endif
