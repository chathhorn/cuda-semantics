#ifndef _KCC_CUDA_H
#define _KCC_CUDA_H
#include <kccSettings.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_datatypes.h>

/* The CUDA Runtime API (abridged). */

#define __CUDA_ARCH__ 200
#define CUDART_VERSION 4020

#define __restrict__ restrict

/* Device Management */
cudaError_t cudaChooseDevice(int* device, const struct cudaDeviceProp* prop);
cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache* pCacheConfig);
cudaError_t cudaDeviceGetLimit(size_t* pValue, enum cudaLimit limit);
cudaError_t cudaDeviceReset(void); /*TODO*/
cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig);
cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaSetDeviceFlags(unsigned int flags);
cudaError_t cudaSetValidDevices(int* device_arr, int len);

/* Error Handling */
const char* cudaGetErrorString(cudaError_t error);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);

/* Stream Management */
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream);

/* Events. */
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags); /*TODO*/
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end); /*TODO*/
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);

/* Execution Control */
// cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
// cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const char* func);
// cudaError_t cudaFuncSetCacheConfig(const char* func, enum cudaFuncCache cacheConfig);
// cudaError_t cudaLaunch(const char *entry);
cudaError_t cudaSetDoubleForDevice(double* d);
cudaError_t cudaSetDoubleForHost(double* d);
// cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset);

/* Memory Management */
cudaError_t cudaFree(void* devPtr);
// cudaError_t cudaFreeArray(struct cudaArray* array);
cudaError_t cudaFreeHost(void* ptr);
// cudaError_t cudaGetSymbolAddress(void** devPtr, const char* symbol);
// cudaError_t cudaGetSymbolSize(size_t* size, const char* symbol);
cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
// cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags);
// cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost);
// cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
// cudaError_t cudaHostUnregister(void* ptr);
cudaError_t cudaMalloc(void** devPtr, size_t size);
// cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);
// cudaError_t cudaMalloc3DArray(struct cudaArray** array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags);
// cudaError_t cudaMallocArray(struct cudaArray** array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height=0, unsigned int flags);
cudaError_t cudaMallocHost(void** ptr, size_t size);
// cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
// cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
// cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray* dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray* src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind);
// cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
// cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, const struct cudaArray* src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
// cudaError_t cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, const struct cudaArray* src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
// cudaError_t cudaMemcpy2DToArray(struct cudaArray* dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
// cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray* dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
// cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms* p);
// cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms* p, cudaStream_t stream);
// cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms* p);
// cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms* p, cudaStream_t stream);
// cudaError_t cudaMemcpyArrayToArray(struct cudaArray* dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray* src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
// cudaError_t cudaMemcpyFromArray(void* dst, const struct cudaArray* src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
// cudaError_t cudaMemcpyFromArrayAsync(void* dst, const struct cudaArray* src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
// cudaError_t cudaMemcpyFromSymbol(void* dst, const char* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind);
// cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const char* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
// cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count);
// cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream);
// cudaError_t cudaMemcpyToArray(struct cudaArray* dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind);
// cudaError_t cudaMemcpyToArrayAsync(struct cudaArray* dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
// cudaError_t cudaMemcpyToSymbol(const char* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind);
// cudaError_t cudaMemcpyToSymbolAsync(const char* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
// cudaError_t cudaMemGetInfo(size_t* free, size_t* total);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
// cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height);
// cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
// cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
// cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream);

/* Unified Addressing */
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes* attributes, void* ptr);

/* Peer Device Memory Access */
cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);
cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags);

/* Version Management */
cudaError_t cudaDriverGetVersion(int* driverVersion);
cudaError_t cudaRuntimeGetVersion(int* runtimeVersion);

/* Memory Fence */
__device__ void __threadfence_block(void);
__device__ void __threadfence(void);
__device__ void __threadfence_system(void);

/* Synchronization */
__device__ void __syncthreads(void);
__device__ int __syncthreads_count(int predicate);
__device__ int __syncthreads_and(int predicate);
__device__ int __syncthreads_or(int predicate);


/* Math */
__host__ __device__ double asin(double x);
__host__ __device__ double atan(double x);
__host__ __device__ double atan2(double x, double y);
__host__ __device__ double cos(double x);
__host__ __device__ double exp(double x);
__host__ __device__ double floor(double x);
__host__ __device__ double fmod(double x, double y);
__host__ __device__ double log(double x);
__host__ __device__ double sin(double x);
__host__ __device__ double sqrt(double x);
__host__ __device__ double tan(double x);

__host__ __device__ float asinf(float x);
__host__ __device__ float atanf(float x);
__host__ __device__ float atan2f(float x, float y);
__host__ __device__ float cosf(float x);
__host__ __device__ float expf(float x);
__host__ __device__ float floorf(float x);
__host__ __device__ float fmodf(float x, float y);
__host__ __device__ float logf(float x);
__host__ __device__ float sinf(float x);
__host__ __device__ float sqrtf(float x);
__host__ __device__ float tanf(float x);

/* Math Intrinsics */
__device__ float __cosf(float x);
__device__ float __expf(float x);
__device__ float __logf(float x);
__device__ float __sinf(float x);
__device__ float __tanf(float x);


#if (__CUDA_ARCH__ >= 200)
/* Dynamic global device memory allocation. */
__host__ __device__ void* malloc(size_t size);
__host__ __device__ void free(void* ptr);

/* Device printf. */
__host__ __device__ int printf(const char* restrict format, ...);

__host__ __device__ void abort(void);
#endif

#endif


