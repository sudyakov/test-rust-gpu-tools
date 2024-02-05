typedef unsigned int uint;
#define GLOBAL
#define KERNEL extern "C" __global__

extern "C" __constant__ int my_constant = 314;

extern "C" __global__ void add(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
      result[i] = a[i] + b[i];
    }
}