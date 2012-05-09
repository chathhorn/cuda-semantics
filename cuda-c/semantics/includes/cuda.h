#ifndef _KCC_CUDA_H
#define _KCC_CUDA_H
#include <kccSettings.h>

enum {
      cudaMemcpyHostToDevice = 1,
      cudaMemcpyDeviceToHost = 2,
};

void cudaMalloc(void**, size_t);
void cudaFree(void*);
void cudaMemcpy(void* dst, void* src, size_t nbytes, int method);
void cudaMemcpyAsync(void* dst, void* src, size_t nbytes, int method, int stream);

void __syncthreads(void);
void cudaStreamSynchronize(int stream);
void cudaDeviceSynchronize(void);

#endif
