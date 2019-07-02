#ifndef _ETH_CNN_
#define _ETH_CNN_

#include "common.h"

//top eth_cnn module in FPGA
void eth_cnn(Dtype *In, Dtype *Params, Dtype *Out);

//void params_mem_rearr(Dtype *In, Dtype *Out_l, Dtype *Out_m, Dtype *Out_s);

void mem_copy(Dtype *In_l, Dtype *Out_l, int Lyr);

#endif // !_ETH_CNN_
