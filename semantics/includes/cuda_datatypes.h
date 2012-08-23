#ifndef _KCC_CUDA_VECTOR_TYPES_H
#define _KCC_CUDA_VECTOR_TYPES_H

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef long long longlong;
typedef unsigned long long ulonglong;

enum cudaMemoryType {
      cudaMemoryTypeHost   = 1,
      cudaMemoryTypeDevice = 2,
};

struct cudaPointerAttributes {
      enum cudaMemoryType memoryType;
      int device;
      void* devicePointer;
      void* hostPointer;
};

enum cudaMemcpyKind {
      cudaMemcpyHostToDevice = 1,
      cudaMemcpyDeviceToHost = 2,
};

typedef enum cudaError {
      cudaSuccess = 0,
      cudaErrorInvalidDevice = 10,
      cudaErrorInvalidResourceHandle = 33,
      cudaErrorNotReady = 34,
      cudaErrorUnsupportedLimit = 42,
      cudaErrorPeerAccessAlready = 50,
      cudaErrorPeerAccessNotEnabled = 51,
} cudaError_t;

enum cudaFuncCache {
      cudaFuncCachePreferNone   = 0,
      cudaFuncCachePreferShared = 1,
      cudaFuncCachePreferL1     = 2,
};

typedef struct dim3 {
      unsigned x, y, z;
} dim3;

struct cudaDeviceProp {
      char name[256];
      size_t totalGlobalMem;
      size_t sharedMemPerBlock;
      int regsPerBlock;
      int warpSize;
      size_t memPitch;
      int maxThreadsPerBlock;
      int maxThreadsDim[3];
      int maxGridSize[3];
      int clockRate;
      size_t totalConstMem;
      int major;
      int minor;
      size_t textureAlignment;
      int deviceOverlap;
      int multiProcessorCount;
      int kernelExecTimeoutEnabled;
      int integrated;
      int canMapHostMemory;
      int computeMode;
      int maxTexture1D;
      int maxTexture2D[2];
      int maxTexture3D[3];
      int maxTexture1DLayered[2];
      int maxTexture2DLayered[3];
      size_t surfaceAlignment;
      int concurrentKernels;
      int ECCEnabled;
      int pciBusID;
      int pciDeviceID;
      int pciDomainID;
      int tccDriver;
      int asyncEngineCount;
      int unifiedAddressing;
      int memoryClockRate;
      int memoryBusWidth;
      int l2CacheSize;
      int maxThreadsPerMultiProcessor;
};

enum cudaLimit {
      cudaLimitStackSize      = 0,
      cudaLimitPrintfFifoSize = 1,
      cudaLimitMallocHeapSize = 2,
};

typedef int cudaStream_t;
typedef int cudaEvent_t;

extern dim3 threadIdx;
extern dim3 blockIdx;
extern dim3 gridDim;
extern dim3 blockDim;
extern int warpSize;

struct cudaPitchedPtr {
      void* ptr;
      size_t pitch;
      size_t xsize;
      size_t ysize;
};

struct cudaExtent {
      size_t width;
      size_t height;
      size_t depth;
};

struct cudaPos {
      size_t x;
      size_t y;
      size_t z;
};

__device__ __host__ struct cudaExtent make_cudaExtent(size_t w, size_t h, size_t d);
__device__ __host__ struct cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz);
__device__ __host__ struct cudaPos make_cudaPos(size_t x, size_t y, size_t z);

/* Built-in Vector Types */

#define DECL_VEC1(T) \
      typedef struct T ## 1 { T x; } T ## 1;
#define DECL_VEC2(T) \
      typedef struct T ## 2 { T x, y; } T ## 2;
#define DECL_VEC3(T) \
      typedef struct T ## 3 { T x, y, z; } T ## 3;
#define DECL_VEC4(T) \
      typedef struct T ## 4 { T x, y, z, w; } T ## 4;

#define DECL_VEC_MAKE1(T) \
      __host__ __device__ T ## 1 make_ ## T ## 1 (T x);
#define DECL_VEC_MAKE2(T) \
      __host__ __device__ T ## 2 make_ ## T ## 2 (T x, T y);
#define DECL_VEC_MAKE3(T) \
      __host__ __device__ T ## 3 make_ ## T ## 3 (T x, T y, T z);
#define DECL_VEC_MAKE4(T) \
      __host__ __device__ T ## 4 make_ ## T ## 4 (T x, T y, T z, T w);

DECL_VEC1(char) DECL_VEC1(uchar)
DECL_VEC2(char) DECL_VEC2(uchar)
DECL_VEC3(char) DECL_VEC3(uchar)
DECL_VEC4(char) DECL_VEC4(uchar)

DECL_VEC1(short) DECL_VEC1(ushort)
DECL_VEC2(short) DECL_VEC2(ushort)
DECL_VEC3(short) DECL_VEC3(ushort)
DECL_VEC4(short) DECL_VEC4(ushort)

DECL_VEC1(int) DECL_VEC1(uint)
DECL_VEC2(int) DECL_VEC2(uint)
DECL_VEC3(int) DECL_VEC3(uint)
DECL_VEC4(int) DECL_VEC4(uint)

DECL_VEC1(long) DECL_VEC1(ulong)
DECL_VEC2(long) DECL_VEC2(ulong)
DECL_VEC3(long) DECL_VEC3(ulong)
DECL_VEC4(long) DECL_VEC4(ulong)

DECL_VEC1(longlong) DECL_VEC1(ulonglong)
DECL_VEC2(longlong) DECL_VEC2(ulonglong)

DECL_VEC1(float)
DECL_VEC2(float)
DECL_VEC3(float)
DECL_VEC4(float)

DECL_VEC1(double)
DECL_VEC2(double)

DECL_VEC_MAKE1(char) DECL_VEC_MAKE1(uchar)
DECL_VEC_MAKE2(char) DECL_VEC_MAKE2(uchar)
DECL_VEC_MAKE3(char) DECL_VEC_MAKE3(uchar)
DECL_VEC_MAKE4(char) DECL_VEC_MAKE4(uchar)

DECL_VEC_MAKE1(short) DECL_VEC_MAKE1(ushort)
DECL_VEC_MAKE2(short) DECL_VEC_MAKE2(ushort)
DECL_VEC_MAKE3(short) DECL_VEC_MAKE3(ushort)
DECL_VEC_MAKE4(short) DECL_VEC_MAKE4(ushort)

DECL_VEC_MAKE1(int) DECL_VEC_MAKE1(uint)
DECL_VEC_MAKE2(int) DECL_VEC_MAKE2(uint)
DECL_VEC_MAKE3(int) DECL_VEC_MAKE3(uint)
DECL_VEC_MAKE4(int) DECL_VEC_MAKE4(uint)

DECL_VEC_MAKE1(long) DECL_VEC_MAKE1(ulong)
DECL_VEC_MAKE2(long) DECL_VEC_MAKE2(ulong)
DECL_VEC_MAKE3(long) DECL_VEC_MAKE3(ulong)
DECL_VEC_MAKE4(long) DECL_VEC_MAKE4(ulong)

DECL_VEC_MAKE1(longlong) DECL_VEC_MAKE1(ulonglong)
DECL_VEC_MAKE2(longlong) DECL_VEC_MAKE2(ulonglong)

DECL_VEC_MAKE1(float)
DECL_VEC_MAKE2(float)
DECL_VEC_MAKE3(float)
DECL_VEC_MAKE4(float)

DECL_VEC_MAKE1(double)
DECL_VEC_MAKE2(double)

#endif
