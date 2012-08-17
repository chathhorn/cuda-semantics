#include <cuda.h>
#include <stdio.h>

int main(int argc, char** argv) {
      int i;

      for (i = 0; i != 64; ++i) {
            printf("%d: %s\n", i, cudaGetErrorString((cudaError_t)i));
      }

      printf("%d: %s\n", 127, cudaGetErrorString((cudaError_t)127));
      printf("%d: %s\n", 10000, cudaGetErrorString((cudaError_t)10000));
}
