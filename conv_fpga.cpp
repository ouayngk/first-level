#include "sds_lib.h"

#include "conv_fpga.h"
#include "common.h"
#include "sub_sample.h"

void conv_fpga(Dtype *In_l,
	Dtype *Params_l,
	Dtype *Out_l,
	int Lyr,
	int InRownum_l,
	int InColnum_l,
	int OutRownum_l,
	int OutColnum_l,
	int Kern,
	int Ichnl,
	int Ochnl,
	int Isec,
	int Osec) {

	int itile = Lyr == 0 ? 1 : Tn;

	conv_l(In_l, Params_l, Out_l, Ichnl, Ochnl, Kern, InRownum_l, InColnum_l, OutRownum_l, OutColnum_l, itile, Isec, Osec, Lyr);

	return;

}

//layer l *********************************************************************************************************************************

void conv_l(
	Dtype *In_l,
	Dtype *Params_l,
	Dtype *Out_l,
	int Ichnl,
	int Ochnl,
	int Kern,
	int InRownum_l,
	int InColnum_l,
	int OutRownum_l,
	int OutColnum_l,
	int IchnlTil,
	int ISec,
	int OSec,
	int Lyr) {

	//set on-chip buffer
	static Dtype b_buf_l[Tm][B_BUF_DEPTH];
#pragma HLS array_partition variable= b_buf_l complete dim=1

	static Dtype out_buf_l[Tm][O_BUF_DEPTH_L];
#pragma HLS array_partition variable= out_buf_l complete dim=1

	//output channel
	for (int n = 0; n < OSec; n++) {
//#pragma HLS dataflow
#pragma HLS loop_tripcount min=1 max=2
		int otile = Lyr == 1 ? ((n == 1) ? 8 : Tm) : Tm;

		if(0 == n){
			conv_bias_read_l(Params_l, b_buf_l, OSec, otile);
		}

		conv_ichnl_l(In_l, Params_l, b_buf_l, out_buf_l, IchnlTil, otile, Ochnl, Kern, n, Lyr, InRownum_l, InColnum_l, OutRownum_l, OutColnum_l, ISec);

		conv_buf_write_l(
			out_buf_l,
			Out_l + n * Tm * OutRownum_l * OutColnum_l,
			otile,
			OutRownum_l,
			OutColnum_l
		);

	}
	return;

}

void conv_ichnl_l(
	Dtype *In_l,
	Dtype *Params_l,
	Dtype Bbuf_l[Tm][B_BUF_DEPTH],
	Dtype OutBuf_l[Tm][O_BUF_DEPTH_L],
	int IchnlTil,
	int OchnlTil,
	int Ochnl,
	int Kern,
	int Ni,
	int Lyr,
	int InRownum_l,
	int InColnum_l,
	int Rownum_l,
	int Colnum_l,
	int ISec
) {
	static Dtype in_buf_l[Tn][I_BUF_DEPTH_L];
#pragma HLS array_partition variable= in_buf_l complete dim=1

	static Dtype w_buf_l[Tn * Tm][W_BUF_DEPTH];
#pragma HLS array_partition variable= w_buf_l complete dim=1

	//input channel
	for (int m = 0; m < ISec; m++) {
#pragma HLS loop_tripcount min=1 max=3

		if (Ni == 0) {
			conv_input_read_l(In_l + (m << ITILESHIFT) * InRownum_l * InColnum_l, in_buf_l, IchnlTil, InRownum_l, InColnum_l, m);
		}

		int otileshift = Lyr == 1 ? ((Ni == 1) ? 3 : OTILESHIFT) : OTILESHIFT;
		int ochnl = Lyr == 1 ? ((Ni == 1) ? 32 : Ochnl) : Ochnl;
		if (Lyr == 2) {
			ochnl = (Ni == 1) ? 48 : Ochnl;
		}
		int real_ochnl = 0;
		switch(Lyr){
		case 0: {real_ochnl = 16; break;}
		case 1: {real_ochnl = 24; break;}
		case 2: {real_ochnl = 32; break;}
		default: {real_ochnl = 0; break;}
		}

		conv_weight_read_l(Params_l + real_ochnl + ((m <<otileshift) * IchnlTil + (Ni << ITILESHIFT) * ochnl) * Kern * Kern, w_buf_l, IchnlTil, OchnlTil, Kern);

		conv_compute_l(
			in_buf_l,
			w_buf_l,
			Bbuf_l,
			OutBuf_l,
			InRownum_l,
			InColnum_l,
			Rownum_l,
			Colnum_l,
			Kern,
			IchnlTil,
			OchnlTil,
			Ni,
			m
		);
	}
	return;
}

void conv_input_read_l(Dtype *In_l, Dtype InBuf_l[Tn][I_BUF_DEPTH_L], int IchnlTil, int InRownum_l, int InColnum_l, int Sec) {
	for (int n = 0; n < IchnlTil; n++) {
#pragma HLS loop_tripcount min=1 max=8
		for (int r = 0; r < InRownum_l; r++) {
#pragma HLS loop_tripcount min=2 max=16
			for (int c = 0; c < InColnum_l; c++) {
#pragma HLS loop_tripcount min=2 max=16
#pragma HLS pipeline
				InBuf_l[n][Sec * InRownum_l * InColnum_l + r * InColnum_l + c] = *In_l++;
			}
		}
	}
	return;
}

void conv_buf_write_l(
	Dtype OutBuf_l[Tm][O_BUF_DEPTH_L],
	Dtype *Out_l,
	int OchnlTil,
	int Rownum_l,
	int Colnum_l
) {
	for (int m = 0; m < OchnlTil; m++) {
#pragma HLS loop_tripcount min=8 max=16
		for (int row = 0; row < Rownum_l; row++) {
#pragma HLS loop_tripcount min=1 max=4
			for (int col = 0; col < Colnum_l; col++) {
#pragma HLS loop_tripcount min=1 max=4
#pragma HLS pipeline
				Dtype data = OutBuf_l[m][row * Colnum_l + col];
				*Out_l++ = (data < 0) ? (Dtype)(0.2 * data) : data;
			}
		}
	}
	return;
}

void conv_compute_l(
	Dtype InBuf_l[Tn][I_BUF_DEPTH_L],
	Dtype WBuf_l[Tn * Tm][W_BUF_DEPTH],
	Dtype BBuf_l[Tm][B_BUF_DEPTH],
	Dtype OutBuf_l[Tm][O_BUF_DEPTH_L],
	int InRownum_l,
	int InColnum_l,
	int Rownum_l,
	int Colnum_l,
	int Kern,
	int IchnlTil,
	int OchnlTil,
	int OSec,
	int ISec
) {
	Dtype partial_l[Tm];
#pragma HLS array_partition variable=pratial_l complete

	for (int k1 = 0; k1 < Kern; k1++) {
#pragma HLS loop_tripcount min=2 max=4
		for (int k2 = 0; k2 < Kern; k2++) {
#pragma HLS loop_tripcount min=2 max=4
			for (int row = 0; row < Rownum_l; row++) {
#pragma HLS loop_tripcount min=1 max=4
				for (int col = 0; col < Colnum_l; col++) {
#pragma HLS loop_tripcount min=1 max=4
#pragma HLS pipeline
					for (int m = 0; m < Tm; m++) {
						if (ISec == 0 && 0 == k1 && 0 == k2) {
							partial_l[m] = ((OchnlTil - 1) < m) ? (Dtype)0.0 : BBuf_l[m][OSec];
						}
						else {
							partial_l[m] = 0.0;
						}
						for (int n = 0; n < Tn; n++) {
							Dtype input = InBuf_l[n][ISec * InRownum_l * InColnum_l + (Kern * row + k1) * InColnum_l + Kern * col + k2];
							Dtype weight = 0;
							if (m < OchnlTil && n < IchnlTil) {
								weight = WBuf_l[m * Tn + n][k1 * Kern + k2];
							}
							partial_l[m] += weight * input;
						}
#pragma HLS dependence variable=OutBuf_l inter false
						if (0 == ISec && 0 == k1 && 0 == k2) {
							OutBuf_l[m][row * Colnum_l + col] = partial_l[m];
						}
						else {
							OutBuf_l[m][row * Colnum_l + col] += partial_l[m];
						}
					}
				}
			}

		}
	}
	return;
}

void conv_bias_read_l(Dtype *Params_l, Dtype Bbuf_l[Tm][B_BUF_DEPTH], int OSec, int OchnlTil) {
	for (int til = 0; til < OSec; til++) {
#pragma HLS loop_tripcount min=1 max=2
		for (int m = 0; m < OchnlTil; m++) {
#pragma HLS pipeline
			Bbuf_l[m][til] = *Params_l++;
		}
	}

	return;
}

void conv_weight_read_l(
	Dtype *Params_l,
	Dtype Wbuf_l[Tn * Tm][W_BUF_DEPTH],
	int IchnlTil,
	int OchnlTil,
	int Kern) {
	for (int m = 0; m < OchnlTil; m++) {
#pragma HLS loop_tripcount min=8 max=16
		for (int n = 0; n < IchnlTil; n++) {
#pragma HLS loop_tripcount min=1 max=8
			for (int k = 0; k < Kern * Kern; k++) {
#pragma HLS loop_tripcount min=4 max=16
#pragma HLS pipeline
				Wbuf_l[m * Tn + n][k] = *Params_l++;
			}
		}
	}
	return;
}



