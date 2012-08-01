// From Appendix B.17 of the CUDA-C Programming Guide.
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

__global__ void mallocTest() {
      char* ptr = (char*)malloc(123);
      printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
      free(ptr);
}

int main() {
      // Set a heap size of 128 megabytes. Note that this must
      // be done before any kernel is launched.
      // TODO
      //cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
      mallocTest<<<1, 5>>>();
      cudaDeviceSynchronize();
      return 0;
}

