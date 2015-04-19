#ifndef BIDIRECTIONALLSTMRNN_H_
#define BIDIRECTIONALLSTMRNN_H_

#include "../matrix/MatrixGenerator.h"
#include "../matrix/GpuMatrix.h"
#include "LstmBlock.h"


class BidirectionalLstmRnn {
	public:
		BidirectionalLstmRnn(MatrixGenerator *W_initializer, MatrixGenerator *R_initializer, MatrixGenerator *p_initializer, MatrixGenerator *logistic_initializer);
		~BidirectionalLstmRnn(void);
		float forwardPropagation(GpuMatrix *sentence);
		void backwardPropagation(GpuMatrix *sentence, int sentence_grammatical);

	private:
		static const int kMaxSentenceLen = 500;
		GpuMatrix *zero_vector;

		// Parameters
		GpuMatrix *Wz_f, *Wz_b;
		GpuMatrix *Rz_f, *Rz_b;
		GpuMatrix *Wi_f, *Wi_b;
		GpuMatrix *Ri_f, *Ri_b;
		GpuMatrix *pi_f, *pi_b;
		GpuMatrix *Wf_f, *Wf_b;
		GpuMatrix *Rf_f, *Rf_b;
		GpuMatrix *pf_f, *pf_b;
		GpuMatrix *Wo_f, *Wo_b;
		GpuMatrix *Ro_f, *Ro_b;
		GpuMatrix *po_f, *po_b;
		GpuMatrix *w_hy_f, *w_hy_b;

		// Buffers for vectorization and theirs columns
		GpuMatrix *pre_z_f, *pre_z_b;
		GpuMatrix *pre_i_f, *pre_i_b;
		GpuMatrix *pre_f_f, *pre_f_b;
		GpuMatrix *c_f, *c_b;
		GpuMatrix *pre_o_f, *pre_o_b;
		GpuMatrix *h_f, *h_b;
		GpuMatrix **pre_z_f_columns, **pre_z_b_columns;
		GpuMatrix **pre_i_f_columns, **pre_i_b_columns;
		GpuMatrix **pre_f_f_columns, **pre_f_b_columns;
		GpuMatrix **c_f_columns, **c_b_columns;
		GpuMatrix **pre_o_f_columns, **pre_o_b_columns;
		GpuMatrix **h_f_columns, **h_b_columns;

		// logistic regression gradients wrt forward and backward features
		GpuMatrix *dL_dh_f, *dL_dh_b;

		// Gates' gradients buffers for vectorization, and theirs columns
		GpuMatrix *dL_dpre_z_f, *dL_dpre_z_b;
		GpuMatrix *dL_dpre_i_f, *dL_dpre_i_b;
		GpuMatrix *dL_dpre_f_f, *dL_dpre_f_b;
		GpuMatrix *dL_dpre_o_f, *dL_dpre_o_b;
		GpuMatrix **dL_dpre_z_f_columns, **dL_dpre_z_b_columns;
		GpuMatrix **dL_dpre_i_f_columns, **dL_dpre_i_b_columns;
		GpuMatrix **dL_dpre_f_f_columns, **dL_dpre_f_b_columns;
		GpuMatrix **dL_dpre_o_f_columns, **dL_dpre_o_b_columns;

		// LSTM blocks
		LstmBlock *forward_lstm_block[BidirectionalLstmRnn::kMaxSentenceLen];
		LstmBlock *backward_lstm_block[BidirectionalLstmRnn::kMaxSentenceLen];

		// parameters gradients
		GpuMatrix *dL_dWz_f, *dL_dWz_b;
		GpuMatrix *dL_dRz_f, *dL_dRz_b;
		GpuMatrix *dL_dWi_f, *dL_dWi_b;
		GpuMatrix *dL_dRi_f, *dL_dRi_b;
		GpuMatrix *dL_dpi_f, *dL_dpi_b;
		GpuMatrix *dL_dWf_f, *dL_dWf_b;
		GpuMatrix *dL_dRf_f, *dL_dRf_b;
		GpuMatrix *dL_dpf_f, *dL_dpf_b;
		GpuMatrix *dL_dWo_f, *dL_dWo_b;
		GpuMatrix *dL_dRo_f, *dL_dRo_b;
		GpuMatrix *dL_dpo_f, *dL_dpo_b;
		GpuMatrix *dL_dw_hy_f, *dL_dw_hy_b;
		GpuMatrix *dL_dx_f, *dL_dx;

		// computation contexts
		GpuMatrixContext *z_f_context, *z_b_context;
		GpuMatrixContext *i_f_context, *i_b_context;
		GpuMatrixContext *f_f_context, *f_b_context;
		GpuMatrixContext *c_f_context, *c_b_context;
		GpuMatrixContext *o_f_context, *o_b_context;
};


#endif
