#ifndef _KCC_CUDA_H
#define _KCC_CUDA_H
#include <kccSettings.h>

enum cudaMemcpyKind {
      cudaMemcpyHostToDevice = 1,
      cudaMemcpyDeviceToHost = 2,
}

typedef enum cudaError {
      cudaSuccess = 0,
      cudaErrorInvalidResourceHandle = 33,
      cudaErrorNotReady = 34,
} cudaError_t;

typedef struct dim3 {
      unsigned x, y, z;
} dim3;

typedef int cudaStream_t;

cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devptr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

cudaError_t cudaDeviceSynchronize(void);

__device__ void __syncthreads(void);
__device__ int __syncthreads_count(int predicate);
__device__ int __syncthreads_and(int predicate);
__device__ int __syncthreads_or(int predicate);

__device__ void __threadfence_block();
__device__ void __threadfence();
__device__ void __threadfence_system();

extern dim3 threadIdx;
extern dim3 blockIdx;
extern dim3 gridDim;
extern dim3 blockDim;

/* Stream management. */

cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
//cudaError_t cudaStreamWaitEvent(cudaStream_t stream);

#endif
