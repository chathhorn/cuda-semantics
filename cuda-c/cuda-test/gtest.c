#include <stdio.h>
#include <cuda.h>

__global__ void gfunc(int a) {
      printf("Success.");
      __syncthreads();
}

__device__ void dfunc(int b) {
      printf("Failure.");
}

__host__ int main(int argc, char** argv) {
      int a = 2, b = 3, c = 903;
      int i;
      gfunc<<<a,b, 0, 2>>>(1);
      cudaStreamSynchronize(2);
}
