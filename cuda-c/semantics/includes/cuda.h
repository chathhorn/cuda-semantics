#ifndef _KCC_CUDA_H
#define _KCC_CUDA_H
#include <kccSettings.h>

enum cudaMemcpyKind {
      cudaMemcpyHostToDevice = 1,
      cudaMemcpyDeviceToHost = 2,
};

typedef enum cudaError {
      cudaSuccess = 0,
      cudaErrorInvalidDevice = 10,
      cudaErrorInvalidResourceHandle = 33,
      cudaErrorNotReady = 34,
      cudaErrorPeerAccessNotEnabled = 50,
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

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned flags); /*TODO*/
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaFreeHost(void* ptr);

cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream);

/* These are nops for now. */
__device__ void __threadfence_block(void); /*TODO*/
__device__ void __threadfence(void); /*TODO*/
__device__ void __threadfence_system(void); /*TODO*/

__device__ void __syncthreads(void);
__device__ int __syncthreads_count(int predicate);
__device__ int __syncthreads_and(int predicate);
__device__ int __syncthreads_or(int predicate);

extern dim3 threadIdx;
extern dim3 blockIdx;
extern dim3 gridDim;
extern dim3 blockDim;
extern int warpSize; /* 32. */

/* Streams. */

cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream);

/* Events. */

cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags); /*TODO*/
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end); /*TODO*/

/* Devices. */

//cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaDeviceReset(void); /*TODO*/
//cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceSynchronize(void);

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice); /* Answer: No. */
cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags);

/* Math. */

__device__ double asin(double x);
__device__ double atan(double x);
__device__ double atan2(double x, double y);
__device__ double cos(double x);
__device__ double exp(double x);
__device__ double floor(double x);
__device__ double fmod(double x, double y);
__device__ double log(double x);
__device__ double sin(double x);
__device__ double sqrt(double x);
__device__ double tan(double x);

__device__ float asinf(float x);
__device__ float atanf(float x);
__device__ float atan2f(float x, float y);
__device__ float cosf(float x);
__device__ float expf(float x);
__device__ float floorf(float x);
__device__ float fmodf(float x, float y);
__device__ float logf(float x);
__device__ float sinf(float x);
__device__ float sqrtf(float x);
__device__ float tanf(float x);

#define __cosf cosf
#define __expf expf
#define __logf logf
#define __sinf sinf
#define __tanf tanf

#endif
