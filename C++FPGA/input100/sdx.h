#ifndef SRC_SDX_TEST_H_
#define SRC_SDX_TEST_H_
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef __SDSCC__
  #include "sds_lib.h"
  #define MEM_ALLOC(size) sds_alloc(size)
  #define MEM_FREE(ptr) sds_free(ptr)
#else
  #define MEM_ALLOC(size) malloc(size)
  #define MEM_FREE(ptr) free(ptr)
#endif

#define TEST_ITR 1000
using namespace std;
/*class perf_counter
{
public:
     unsigned long long int tot, cnt, calls;
     perf_counter() : tot(0), cnt(0), calls(0) {};
     inline void reset() { tot = cnt = calls = 0; }
     inline void start() { cnt = sds_clock_counter(); calls++; };
     inline void stop() { tot += (sds_clock_counter() - cnt); };
     inline uint64_t avg_cpu_cycles() { return (tot / calls); };
};
*/
int check(float* arr1, float* arr2, int N){

	for(int i=0;i<N;i++){
		float t1 = arr1[i];
		float t2 = arr2[i];
		if(t1!=t2){
			std::cout<<i<<" Error : HW and SW are not matched"<<std::endl;
			return 1;
		}

	}

	return 0;
}


#endif /* SRC_SDX_TEST_H_ */
