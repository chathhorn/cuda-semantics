// From Appendix B.17 of the CUDA-C Programming Guide.

#include <stdlib.h>
#include <stdio.h>

#define NUM_BLOCKS 20

__device__ int* dataptr[NUM_BLOCKS]; // Per-block pointer

__global__ void allocmem() {
      // Only the first thread in the block does the allocation
      // since we want only one allocation per block.
      if (threadIdx.x == 0)
            dataptr[blockIdx.x] = (int*)malloc(blockDim.x * 4);
      __syncthreads();
      // Check for failure
      if (dataptr[blockIdx.x] == NULL)
            return;
      // Zero the data with all threads in parallel
      dataptr[blockIdx.x][threadIdx.x] = 0;
}

// Simple example: store thread ID into each element
__global__ void usemem() {
      int* ptr = dataptr[blockIdx.x];
      if (ptr != NULL)
            ptr[threadIdx.x] += threadIdx.x;
}

// Print the content of the buffer before freeing it
__global__ void freemem() {
      int* ptr = dataptr[blockIdx.x];
      if (ptr != NULL)
            printf("Block %d, Thread %d: final value = %d\n",
                        blockIdx.x, threadIdx.x, ptr[threadIdx.x]);
      // Only free from one thread!
      if (threadIdx.x == 0)
            free(ptr);
}

int main() {
      cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
      // Allocate memory
      allocmem<<< NUM_BLOCKS, 10 >>>();
      // Use memory
      usemem<<< NUM_BLOCKS, 10 >>>();
      usemem<<< NUM_BLOCKS, 10 >>>();
      usemem<<< NUM_BLOCKS, 10 >>>();
      // Free memory
      freemem<<< NUM_BLOCKS, 10 >>>();
      cudaDeviceSynchronize();
      return 0;
}

