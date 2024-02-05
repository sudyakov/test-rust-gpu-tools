typedef unsigned int uint;

#include <stdio.h> 
#include <stdint.h> 

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

#include <cuda_runtime.h> 
#include <helper_cuda.h>
  
#define DLLEXPORT __declspec(dllexport) 
#define MAX_OUTPUT_RESULTS 32 
#if (__CUDACC_VER_MAJOR__ >= 10) && (__CUDA_ARCH__ > 300) 
#define ROTR(v, n) __funnelshift_rc((v), (v), n) 
#else
#define ROTR(v, n) ((v) >> n) | ((v) << (32 - n))
#endif

#define GLOBAL
#define KERNEL extern "C" __global__

KERNEL void sortDescending(uint num, GLOBAL uint *data, GLOBAL uint *result) {

  for (int i = 0; i < num; i++) {
    result[i] = data[i];
  }

  thrust::sort(thrust::device, result, result + num, thrust::greater<uint>());
}