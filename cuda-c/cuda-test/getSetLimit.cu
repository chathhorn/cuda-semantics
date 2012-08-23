#include <cuda.h>
#include <stdio.h>

int main(int argc, char** argv) {

      size_t limit = 0;

      cudaDeviceGetLimit(&limit, cudaLimitStackSize);
      printf("cudaLimitStackSize: %u\n", (unsigned)limit);
      cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
      printf("cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
      cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
      printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);

      limit = 9999;
      
      cudaDeviceSetLimit(cudaLimitStackSize, limit);
      cudaDeviceSetLimit(cudaLimitPrintfFifoSize, limit);
      cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit);

      limit = 0;

      cudaDeviceGetLimit(&limit, cudaLimitStackSize);
      printf("New cudaLimitStackSize: %u\n", (unsigned)limit);
      cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
      printf("New cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
      cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
      printf("New cudaLimitMallocHeapSize: %u\n", (unsigned)limit);

      return 0;
}
