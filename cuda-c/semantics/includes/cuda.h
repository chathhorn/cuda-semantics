#ifndef _KCC_CUDA_H
#define _KCC_CUDA_H
#include <kccSettings.h>

// Hack to partly support C++-style initialization of structures because nvcc
// doesn't support the C style (only for dim3?)...
#define dim3(x, y, z) {x, y, z}

enum cudaMemcpyKind {
      cudaMemcpyHostToDevice = 1,
      cudaMemcpyDeviceToHost = 2,
};

typedef enum cudaError {
      cudaSuccess = 0,
      cudaErrorInvalidResourceHandle = 33,
      cudaErrorNotReady = 34,
} cudaError_t;

typedef struct dim3 {
      unsigned x, y, z;
} dim3;

typedef int cudaStream_t;
typedef int cudaEvent_t;

/* Memory. */

cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devptr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

/* Flags are ignored. */
cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned flags); 
//cudaError_t cudaMallocHost(void** ptr, size_t size);
//cudaError_t cudaFreeHost(void* ptr);

/* These are nops for now. */
__device__ void __threadfence_block();
__device__ void __threadfence();
__device__ void __threadfence_system();

cudaError_t cudaDeviceSynchronize(void);

__device__ void __syncthreads(void);
__device__ int __syncthreads_count(int predicate);
__device__ int __syncthreads_and(int predicate);
__device__ int __syncthreads_or(int predicate);

extern dim3 threadIdx;
extern dim3 blockIdx;
extern dim3 gridDim;
extern dim3 blockDim;
extern int warpSize;

/* Streams. */

cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream);

/* Events. */

cudaError_t cudaEventCreate(cudaEvent_t* event);
/* Flags are ignored. */
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
//cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);

#endif
