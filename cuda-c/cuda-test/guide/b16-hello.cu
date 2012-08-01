// From Appendix B.16 of the CUDA-C Programming Guide.

#include "stdio.h"
// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

__global__ void helloCUDA(float f) {
      printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

int main() {
      helloCUDA<<<1, 5>>>(1.2345f);
      cudaDeviceReset();
      return 0;
}

