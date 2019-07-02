#ifndef _COMMON_H_
#define _COMMON_H_

//#include "hls_half.h"

//data type
typedef float Dtype;

#define IMG_W 64
#define IMG_H 64
#define CONV_LAYER_NUM 3
#define FC_LAYER_NUM 3
#define CLASS_NUM 32

//input channel tile size
#define Tn 8
#define ITILESHIFT 3
//output channel tile size
#define Tm 16
#define OTILESHIFT 4

#define I_BUF_DEPTH_L 16*16

#define W_BUF_DEPTH 16

#define O_BUF_DEPTH_L 4*4

#define B_BUF_DEPTH 2


#endif // !_COMMON_H_
