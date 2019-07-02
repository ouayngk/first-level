#ifndef __SUB_SAMPLE__
#define __SUB_SAMPLE__

#include "common.h"

void sub_sample(Dtype *In, Dtype *Out_l);

void sub_sample_l(Dtype In[64][64], Dtype Sub_l[16][16]);

void sub_buf_write(Dtype Sub_l[16][16], Dtype *Out_l);

#endif // !__SUB_SAMPLE__

