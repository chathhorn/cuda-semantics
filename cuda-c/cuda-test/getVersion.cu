#include <cuda.h>
#include <stdio.h>

int main(int argc, char** argv) {
      int driver_version = 0, runtime_version = 0;

      cudaDriverGetVersion(&driver_version);
      cudaRuntimeGetVersion(&runtime_version);

      printf("Driver Version: %d\n"
             "Runtime Version: %d\n",
             driver_version, runtime_version);

      return 0;
}
