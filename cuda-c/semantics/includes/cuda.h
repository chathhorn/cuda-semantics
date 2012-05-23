#ifndef _KCC_CUDA_H
#define _KCC_CUDA_H
#include <kccSettings.h>

typedef enum cudaMemcpyKind {
      cudaMemcpyHostToDevice = 1,
      cudaMemcpyDeviceToHost = 2,
};

typedef enum cudaError {
      cudaSuccess = 0,
} cudaError_t;

typedef int cudaStream_t;

cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devptr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaDeviceSynchronize(void);

void __syncthreads(void);

extern int threadIdx;
extern int blockIdx;
extern int gridDim;
extern int blockDim;

#endif
