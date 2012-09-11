// Assert requires compute capability 2.x or higher
// (e.g., "nvcc -arch=sm_21").
#include <assert.h> 
#include <stdio.h> 
#include <cuda.h> 

#define N 10

__global__ void synctest(void) {
      int x, tid = threadIdx.x;

      x = __syncthreads_count(tid % 2 == 0);

      assert(x == N/2 + !!(N % 2));

      x = __syncthreads_count(tid % 3 == 0);

      assert(x == N/3 + !!(N % 3));

      x = __syncthreads_and(1);

      assert(x);

      x = __syncthreads_and(tid != 0);

      assert(!x);

      x = __syncthreads_and(0);

      assert(!x);

      x = __syncthreads_or(1);

      assert(x);

      x = __syncthreads_or(tid != 0);

      assert(x);

      x = __syncthreads_or(0);

      assert(!x);
}

int main(void) { 
      cudaError_t e;
      synctest<<<1, N>>>(); 
      e = cudaDeviceSynchronize();

      if (e) printf("Error: %s\n", cudaGetErrorString(e));
      else printf("PASS\n");
}

