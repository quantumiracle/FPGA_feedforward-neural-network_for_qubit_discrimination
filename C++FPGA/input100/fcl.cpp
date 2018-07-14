#include "fcl.h"



float relu(float x){
#pragma HLS inline
	return x>0 ? x : 0;
}






void fcl_1(const int input[100],
		const float weights[100*240],
		const float bias[240],
		float output[240]
		)
{
//unsigned long long __sds_est_clk_start, __sds_est_clk_end;
//sds_est_clk_start = sds_clock_counter();


	float IBRAM[100];
	float WBRAM[100][240];
	float biasBRAM[240];
	float OBRAM[240];
#pragma HLS array_partition variable=WBRAM complete dim=0
#pragma HLS array_partition variable=biasBRAM complete dim=0
#pragma HLS array_partition variable=OBRAM complete dim=0


			//copy_input :
			for(int j=0;j<100;j++){
#pragma HLS unroll
				IBRAM[j] = input[j];
			}


	// load weights & bias at first iteration only
	if(1){


			for(int j=0;j<100;j++){
				for(int k=0;k<240;k++){
				#pragma HLS unroll
					WBRAM[j][k] = weights[j*240+k];
				}
			}



		//copy_bias:
		for(int i=0;i<240;i++){
#pragma HLS unroll
			biasBRAM[i] = bias[i];
		}

	}

	//////////////////////////////////////////////////////////////////////
	//						    FC 							//
	//////////////////////////////////////////////////////////////////////

//ROW:
for (int i=0;i<240;i++){
	//COL:
	OBRAM[i]=0;
	for (int j=0;j<100;j++){

#pragma HLS PIPELINE
		OBRAM[i]+=IBRAM[j]*WBRAM[j][i];
	}
}
		//copy_output:

				for(int k=0;k<240;k++){
	#pragma HLS unroll
					output[k]=relu(OBRAM[k]+biasBRAM[k]);
				}





}







void fcl_2(
		const float input[240],
		const float weights[240*2],
		const float bias[2],
		float output[2]
		)
{
  //unsigned long long __sds_est_clk_start, __sds_est_clk_end;
  //__sds_est_clk_start = sds_clock_counter();


	float IBRAM[240];
	float WBRAM[240][2];
	float biasBRAM[2];
	float OBRAM[2];
#pragma HLS array_partition variable=WBRAM complete dim=1
#pragma HLS array_partition variable=biasBRAM complete dim=0
#pragma HLS array_partition variable=OBRAM complete dim=0


			//copy_input :
			for(int j=0;j<240;j++){
#pragma HLS unroll
				IBRAM[j] = input[j];
			}


	// load weights & bias at first iteration only
	if(1){

			//copy_kernel_1:
			for(int j=0;j<240;j++){
				for(int k=0;k<2;k++){
#pragma HLS unroll
					WBRAM[j][k] = weights[j*2+k];
				}
			}



		//copy_bias:
		for(int i=0;i<2;i++){
#pragma HLS unroll
			biasBRAM[i] = bias[i];
		}

	}

	//////////////////////////////////////////////////////////////////////
	//						    FC 							//
	//////////////////////////////////////////////////////////////////////

//ROW:
for (int i=0;i<2;i++){
	//COL:
	OBRAM[i]=0;
	for (int j=0;j<240;j++){

#pragma HLS PIPELINE II=1
		OBRAM[i]+=IBRAM[j]*WBRAM[j][i];
	}
}
		//copy_output:

				for(int k=0;k<2;k++){
#pragma HLS unroll
					output[k]=relu(OBRAM[k]+biasBRAM[k]);
				}



}
