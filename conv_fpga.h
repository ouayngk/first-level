#ifndef _CONV_FPGA_H_
#define _CONV_FPGA_H_

#include "common.h"

//conv in fpga;include three modules
#pragma SDS data access_pattern(In_l:SEQUENTIAL, Params_l:SEQUENTIAL, Out_l:SEQUENTIAL)
#pragma SDS data mem_attribute(In_l:PHYSICAL_CONTIGUOUS, Params_l:PHYSICAL_CONTIGUOUS, Out_l:PHYSICAL_CONTIGUOUS)
#pragma SDS data copy(In_l[0:Ichnl*InRownum_l*InColnum_l], Params_l[0:Ochnl+Ichnl*Ochnl*Kern*Kern], Out_l[0:Ochnl*OutRownum_l*OutColnum_l])
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
	int Osec);

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
	int Lyr);

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
);

void conv_input_read_l(Dtype *In_l, Dtype InBuf_l[Tn][I_BUF_DEPTH_L], int IchnlTil, int InRownum_l, int InColnum_l, int Sec);

void conv_buf_write_l(
	Dtype OutBuf_l[Tm][O_BUF_DEPTH_L],
	Dtype *Out_l,
	int OchnlTil,
	int Rownum_l,
	int Colnum_l
);

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
);

void conv_bias_read_l(Dtype *Params_l, Dtype Bbuf_l[Tm][B_BUF_DEPTH], int OSec, int OchnlTil);

void conv_weight_read_l(
	Dtype *Params_l,
	Dtype Wbuf_l[Tn * Tm][W_BUF_DEPTH],
	int IchnlTil,
	int OchnlTil,
	int Kern);


#endif // !_CONV_FPGA_H_
