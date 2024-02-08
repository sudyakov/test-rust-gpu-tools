
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

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

#define GLOBAL

__global__
void sort(uint num, GLOBAL uint *data, GLOBAL uint *result) {

  for (int i = 0; i < num; i++) {
    result[i] = data[i];
  }
  thrust::sort(thrust::device, result, result + num, thrust::greater<uint>());
}


extern "C" {
__host__ DLLEXPORT void
sortDescending(const uint32_t dimgrid, const uint32_t threads, uint num, GLOBAL uint *data, GLOBAL uint *result)
{
  sort<<<dimgrid, threads>>>(num, data, result);
}
}
