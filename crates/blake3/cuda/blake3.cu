typedef unsigned int uint;
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <cuda_runtime.h> 

#include <stdio.h> 
#include <stdint.h>

#if defined(_WIN32)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif /* _WIN32 */

#define MAX_OUTPUT_RESULTS 32

// Written and optimized by Dave Collins Sep 2023.
#if (__CUDACC_VER_MAJOR__ >= 10) && (__CUDA_ARCH__ > 300)
#define ROTR(v, n) __funnelshift_rc((v), (v), n)
#else
#define ROTR(v, n) ((v) >> n) | ((v) << (32 - n))
#endif

// dim3 blocks(64);
// dim3 threads(256); 
// __global__ void sort(){
//   printf("sort()");
// }

#define GLOBAL
#define KERNEL extern "C" __global__
KERNEL void sortDescending(const dim3 dimgrid, const dim3 threads, uint num, GLOBAL uint *data, GLOBAL uint *result) {

  for (int i = 0; i < num; i++) {
    result[i] = data[i];
  }


//sort();
 //sort<<<dimgrid, threads>>>();
 thrust::sort(thrust::device, result, result + num, thrust::greater<uint>());

};


