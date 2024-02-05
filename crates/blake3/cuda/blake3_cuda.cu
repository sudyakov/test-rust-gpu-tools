//Cuda program to add two vectors
typedef unsigned int uint;
  /**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */
#include <stdio.h> // printf
#include <stdint.h> // uint32_t

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h> // cudaMemcpy, cudaMemcpyToSymbol, etc.
#include <helper_cuda.h> // helper function CUDA error checking and initialization
  
#define DLLEXPORT __declspec(dllexport) // Определение для экспорта функций из dll в Windows 
#define MAX_OUTPUT_RESULTS 32 // Максимальное количество результатов для вывода
#if (__CUDACC_VER_MAJOR__ >= 10) && (__CUDA_ARCH__ > 300) 
#define ROTR(v, n) __funnelshift_rc((v), (v), n) // Смешение битов вправо с использованием встроенной функции сдвига для новых версий CUDA
#else
#define ROTR(v, n) ((v) >> n) | ((v) << (32 - n)) // Смешение битов вправо с использованием битовых операций для старых версий CUDA
#endif

// Определение для глобальных и статических функций.
#define GLOBAL
#define KERNEL extern "C" __global__

KERNEL void sortDescending(int* data, int n) {
 printf("sortDescending: Hello, World!\n");
}

KERNEL void sortAscending(int* data, int n) {
 printf("sortAscending: Hello, World!\n");

}

extern "C" {
 __global__
void hello() {
      // Вывод на екран сообщения
    printf("hello: Hello, World!\n");

//   int threads_per_block = 256;
//   int num_blocks = (n + threads_per_block - 1) / threads_per_block;

//   // вызов ядра
//   sortDescending<<<num_blocks, threads_per_block>>>(data, n);
  
//   // синхронизация
//   cudaDeviceSynchronize();
  }
}