#include "ap_int.h"

float relu(float x);

#pragma SDS data mem_attribute(input:PHYSICAL_CONTIGUOUS,weights:PHYSICAL_CONTIGUOUS,bias:PHYSICAL_CONTIGUOUS,output:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(input:SEQUENTIAL,weights:SEQUENTIAL,bias:SEQUENTIAL,output:SEQUENTIAL)
#pragma SDS data zero_copy(input,weights,bias,output)
void fcl_1(const int input[100],
		const float weights[100*240],
		const float bias[240],
		float output[240]
		);
#pragma SDS data mem_attribute(input:PHYSICAL_CONTIGUOUS,weights:PHYSICAL_CONTIGUOUS,bias:PHYSICAL_CONTIGUOUS,output:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(input:SEQUENTIAL,weights:SEQUENTIAL,bias:SEQUENTIAL,output:SEQUENTIAL)
#pragma SDS data zero_copy(input,weights,bias,output)
////#pragma SDS data copy(weights[0:kernel_size],bias[0:bias_size])
void fcl_2(const float input[240],
		const float weights[240*2],
		const float bias[2],
		float output[2]
		);
