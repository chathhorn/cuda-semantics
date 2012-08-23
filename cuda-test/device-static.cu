#include <cuda.h>

__device__ int bss_device;
int bss_host;

__global__ void pass() {
      printf("Zero: %i\n", bss_device);
}

__global__ void fail() {
      printf("Zero: %i\n", bss_host);
}

int main(int argc, char** argv) {
      pass<<<1, 1>>>();
      fail<<<1, 1>>>();
      cudaDeviceSynchronize();

      return 0;
}


