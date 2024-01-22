// CUDA
#ifdef __CUDACC__
  #define GLOBAL
  #define KERNEL extern "C" __global__


KERNEL void add(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
      result[i] = a[i] + b[i];
    }
}