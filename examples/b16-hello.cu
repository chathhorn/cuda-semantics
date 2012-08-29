// From Appendix B.16 of the CUDA-C Programming Guide.

#include "stdio.h"
#include "cuda.h"

__global__ void helloCUDA(float f) {
      printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

int main() {
      helloCUDA<<<1, 5>>>(1.2345f);
      cudaDeviceReset();
      return 0;
}

