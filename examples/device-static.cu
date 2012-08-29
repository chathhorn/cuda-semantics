#include <cuda.h>
#include <stdio.h>

__device__ int device;
int host;

__global__ void pass() {
      printf("Zero: %i\n", device);
}

__global__ void fail() {
      // This should halt with an access error.
      printf("Zero: %i\n", host);
}

int main(int argc, char** argv) {
      pass<<<1, 1>>>();
      fail<<<1, 1>>>();
      cudaDeviceSynchronize();

      return 0;
}


