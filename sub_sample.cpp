#include "sds_lib.h"

#include "sub_sample.h"
#include "common.h"

//sub_sample in FPGA

void sub_sample(Dtype *In, Dtype *Out_l) {

	Dtype sub_input_buffer[64][64];
	Dtype sub_l[16][16];

	for (int row = 0; row < 64; row++) {
		for (int col = 0; col < 64; col++) {
			sub_input_buffer[row][col] = 1.0 / 255.0 * In[row * 64 + col];
		}
	}

	sub_sample_l(sub_input_buffer, sub_l);

	sub_buf_write(sub_l, Out_l);

}

void sub_sample_l(Dtype In[64][64], Dtype Sub_l[16][16]) {
	Dtype value = 0;
	Dtype x_mean_reduce_64 = 0;
	for (int row = 0; row < 16; row++) {
		for (int col = 0; col < 16; col++) {
			value = In[row * 4][col * 4] + In[row * 4][col * 4 + 1] + In[row * 4][col * 4 + 2] + In[row * 4][col * 4 + 3]
				+ In[row * 4 + 1][col * 4] + In[row * 4 + 1][col * 4 + 1] + In[row * 4 + 1][col * 4 + 2] + In[row * 4 + 1][col * 4 + 3]
				+ In[row * 4 + 2][col * 4] + In[row * 4 + 2][col * 4 + 1] + In[row * 4 + 2][col * 4 + 2] + In[row * 4 + 2][col * 4 + 3]
				+ In[row * 4 + 3][col * 4] + In[row * 4 + 3][col * 4 + 1] + In[row * 4 + 3][col * 4 + 2] + In[row * 4 + 3][col * 4 + 3];
			Sub_l[row][col] = value / 16.0;
		}
	}

	for (int row = 0; row < 16; row++) {
		for (int col = 0; col < 16; col++) {
			x_mean_reduce_64 += Sub_l[row][col] / 256.0;
		}
	}

	for (int row = 0; row < 16; row++) {
		for (int col = 0; col < 16; col++) {
			Sub_l[row][col] = Sub_l[row][col] - x_mean_reduce_64;
		}
	}
}


void sub_buf_write(Dtype Sub_l[16][16], Dtype *Out_l) {
	for (int row = 0; row < 16; row++) {
		for (int col = 0; col < 16; col++) {
			*Out_l++ = Sub_l[row][col];
		}
	}
}
