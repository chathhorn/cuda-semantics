// Based on: https://gist.github.com/1392067

#include <cuda.h>
#include <stdio.h>

#define NBLOCKS 4
#define NTHREADS 4

#define N (NTHREADS * NBLOCKS)
#define NBYTES (N * sizeof(unsigned))

#define SWAP(a, b) { unsigned tmp = (a); (a) = (b); (b) = tmp; }

__global__ void bitonic_sort_step(unsigned* values, unsigned j, unsigned k) {
      unsigned tid, ixj;
      tid = threadIdx.x + blockDim.x * blockIdx.x;

      /* The threads with the lowest ids sort the array. */
      ixj = tid ^ j;
      if (ixj > tid) {
            if (tid & k) {
                  /* Sort descending */
                  if (values[tid] < values[ixj]) {
                        /* exchange(tid,ixj); */
                        SWAP(values[tid], values[ixj]);
                  }
            } else {
                  /* Sort ascending */
                  if (values[tid] > values[ixj]) {
                        /* exchange(tid,ixj); */
                        SWAP(values[tid], values[ixj]);
                  }
            }
      }
}

int main(int argc, char** argv) {
      unsigned host[N] = {
            5, 19, 3, 15, 2, 0, 1, 6, 8, 7, 11, 10, 13, 12, 4, 4
      };
      unsigned* device;
      unsigned i, j, k;

      puts("Before sort:");
      for (i = 0; i != N; ++i) {
            printf("%u ", host[i]);
      }
      puts("");

      cudaMalloc(&device, NBYTES);

      cudaMemcpy(device, host, NBYTES, cudaMemcpyHostToDevice);

      /* Major step */
      for (k = 2; k <= N; k <<= 1) {
            /* Minor step */
            for (j = k >> 1; j > 0; j >>= 1) {
                  bitonic_sort_step<<<NBLOCKS, NTHREADS>>>(device, j, k);
            }
      }

      cudaMemcpy(host, device, NBYTES, cudaMemcpyDeviceToHost);
      cudaFree(device);

      puts("After sort:");
      for (i = 0; i != N; ++i) {
            printf("%u ", host[i]);
      }
      puts("");

      return 0;
}
