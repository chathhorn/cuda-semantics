// From Appendix B.17 of the CUDA-C Programming Guide.

#include <stdlib.h>

__global__ void mallocTest() {
      __shared__ int* data;
      // The first thread in the block does the allocation
      // and then shares the pointer with all other threads
      // through shared memory, so that access can easily be
      // coalesced. 64 bytes per thread are allocated.
      if (threadIdx.x == 0)
            data = (int*)malloc(blockDim.x * 64);
      __syncthreads();
      // Check for failure
      if (data == NULL)
            return;
      // Threads index into the memory, ensuring coalescence
      int* ptr = data;
      for (int i = 0; i < 64; ++i)
            ptr[i * blockDim.x + threadIdx.x] = threadIdx.x;
      // Ensure all threads complete before freeing
      __syncthreads();
      // Only one thread may free the memory!
      if (threadIdx.x == 0)
            free(data);
}

int main() {
      cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
      mallocTest<<<10, 128>>>();
      cudaDeviceSynchronize();
      return 0;
}

