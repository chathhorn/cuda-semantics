#include <cuda.h>
#include <stdio.h>

int main(int argc, char** argv) {
      cudaError_t e;

      e = cudaPointerGetAttributes((struct cudaPointerAttributes*) 0, (void*) 0);

      printf("Error: %d\n", e);

      return 0;
}
