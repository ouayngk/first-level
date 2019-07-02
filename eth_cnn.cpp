#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <string.h>

#include "sds_lib.h"
///#include "hls_half.h"

#include "get_param.h"
#include "check.h"
#include "eth_cnn.h"
#include "check.h"
#include "common.h"
#include "conv_fpga.h"
#include "performance.h"
#include "sub_sample.h"

//shape of feature map in each layer
extern const int SHAPE_l[4];
//channel number in each layer
extern const int CHNEL_l[8];
//kernel size in each layer
extern const int KERNL[3];

void eth_cnn(Dtype *In, Dtype *Out, Dtype *Params) {

	std::cout << " [INFO] " << __FUNCTION__ << " , " << __LINE__ <<
		": Start ETH_CNN in FPGA..." << std::endl;

	//set three params off-chip memory
	Dtype *params_l = (Dtype *)sds_alloc(180250 * sizeof(Dtype));
	mem_check(params_l);
	//buf size should meet maxium memory requirment for each layer

	int max_buf_l = 32 * 16 * 16;
	Dtype *buffer_l_A = (Dtype *)sds_alloc(max_buf_l * sizeof(Dtype));
	Dtype *buffer_l_B = (Dtype *)sds_alloc(max_buf_l * sizeof(Dtype));

	mem_check(buffer_l_A);
	mem_check(buffer_l_B);

	Dtype *cur_params_l = Params;

	float total_time = 0;

	for (int c_layer = 0; c_layer < CONV_LAYER_NUM; c_layer++) {
		int input_col_num_l = SHAPE_l[c_layer];
		int input_row_num_l = SHAPE_l[c_layer];
		int output_col_num_l = SHAPE_l[c_layer + 1];
		int output_row_num_l = SHAPE_l[c_layer + 1];

		if (0 == c_layer) {
			std::cout << "[INFO] " << __FUNCTION__ << ", " << __LINE__ <<
				": " << c_layer << "th conv layer." << std::endl;
			int isec = 1;
			int osec = 1;

			sub_sample(In, buffer_l_A);
			perf_counter perf;
			perf.start();
			conv_fpga(buffer_l_A,
				      cur_params_l, 
				      buffer_l_B,
				      c_layer, 
				      input_row_num_l,
				      input_col_num_l,
				      output_row_num_l, 
				      output_col_num_l,
				      KERNL[c_layer],
				      CHNEL_l[c_layer],
			          CHNEL_l[c_layer + 1],
			          isec,
				      osec);
			perf.stop();
			uint64_t cpu_cycles = perf.avg_cpu_cycles();
			float lyr_time = (float)cpu_cycles / 1.5e9;
			total_time += lyr_time;
			std::cout << "[INFO] " << __FUNCTION__ << "' " << __LINE__ <<
				": Finish in " << lyr_time << "s. " << std::endl;

			//update pointer to parameters
			cur_params_l += (CHNEL_l[0] * CHNEL_l[1] * KERNL[0] * KERNL[0] + CHNEL_l[1]);
		}
		else {
			std::cout << "[INFO] " << __FUNCTION__ << ", " << __LINE__ <<
				": " << c_layer << "th conv layer." << std::endl;
			int isec = CHNEL_l[c_layer] / Tn;
			int osec = 2;
			mem_copy(buffer_l_B, buffer_l_A, c_layer);
			perf_counter perf;
			perf.start();
			conv_fpga(buffer_l_A,
				      cur_params_l,
				      buffer_l_B,
				      c_layer,
				      input_row_num_l,
				      input_col_num_l,
				      output_row_num_l,
				      output_col_num_l,
				      KERNL[c_layer],
				      CHNEL_l[c_layer],
			          CHNEL_l[c_layer + 1],
			          isec,
				      osec);
			perf.stop();
			uint64_t cpu_cycles = perf.avg_cpu_cycles();
			float lyr_time = (float)cpu_cycles / 1.5e9;
			total_time += lyr_time;
			std::cout << "[INFO] " << __FUNCTION__ << "' " << __LINE__ <<
				": Finish in " << lyr_time << "s. " << std::endl;

			cur_params_l += (CHNEL_l[c_layer] * CHNEL_l[c_layer + 1] * KERNL[c_layer] * KERNL[c_layer] + CHNEL_l[c_layer + 1]);
		}
	}
	std::cout << "[INFO] " << __FUNCTION__ << ", " << __LINE__ <<
		": Total time in conv layer, " << total_time << "s. " << std::endl;

	for(int i = 0; i < 32; i++){
		*Out++ = buffer_l_B[i];
	};

	//free memory
	sds_free(buffer_l_A);
	sds_free(buffer_l_B);
	sds_free(params_l);

	return;
}

void mem_copy(Dtype *In_l, Dtype *Out_l, int Lyr) {
	int row_num_l = SHAPE_l[Lyr];
	int col_num_l = SHAPE_l[Lyr];
	int chnl_l = CHNEL_l[Lyr];

	int copy_num_l = chnl_l * row_num_l * col_num_l;

	memcpy(Out_l, In_l, copy_num_l * sizeof(Dtype));
}
