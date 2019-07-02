#include <fstream>
#include <iostream>
#include "get_param.h"

extern const int KERNL[];
extern const int CHNEL_l[];
extern const int CHNEL_m[];
extern const int CHNEL_s[];

int param_size() {
	int size = (CHNEL_l[0] * CHNEL_l[1] * KERNL[0] * KERNL[0] + CHNEL_l[1]); //3*(16*4*4)

	//conv layer
	for (int c_layer = 1; c_layer < CONV_LAYER_NUM; c_layer++) {
		//weights
		size += (CHNEL_l[c_layer] * CHNEL_l[c_layer + 1] * KERNL[c_layer] * KERNL[c_layer]);
		//bias
		size += (CHNEL_l[c_layer + 1]);
	}
	return size;
}

void get_params(Dtype *Params, int ReadSize) {
	std::cout << "[INFO] " << __FUNCTION__ << ", " << __LINE__ <<
		":load parmas from file" << std::endl;
	std::ifstream params_bin("/mnt/Params.bin", std::ios::binary);
	char *in_buf = reinterpret_cast<char *>(Params);
	params_bin.read(in_buf, ReadSize * sizeof(Dtype));
	params_bin.close();
	return;
}
