#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

#define N 8
// NRUNS should be < N.
#define NRUNS 2
#define NBLOCKS 2
// NTHREADS*NBLOCKS should equal N
#define NTHREADS (N/NBLOCKS)
// Also, NBLOCKS should equal NTHREADS.

// Sums an array.
__global__ void sum(int* g_idata, int* g_odata) {
      extern __shared__ int shared[];
      int i, tid = threadIdx.x;

      shared[tid] = g_idata[blockIdx.x * blockDim.x + tid];
      
      __syncthreads();

      if (tid < blockDim.x/2) {
            shared[tid] += shared[blockDim.x/2 + tid];
      }

      //__syncthreads();

      if (tid == 0) {
            for (i = 1; i != blockDim.x/2 + !!(blockDim.x % 2); ++i) {
                  shared[0] += shared[i];
            }

            g_odata[blockIdx.x] = shared[0];
      }
}

int main(int argc, char** argv) {
      int* d_idata, *d_odata, *d_scratch;
      int i;
      int h_data[N];

      // Use a different stream for every run. 
      cudaStream_t streams[NRUNS];

      printf("INPUT: ");
      for(i = 0; i != N; ++i) {
            h_data[i] = (11 + i * i) % 7;
            printf(" %d ", h_data[i]);
      }
      printf("\n");

      cudaMalloc(&d_scratch, NBLOCKS * NRUNS * sizeof(int));
      cudaMalloc(&d_idata, N * sizeof(int));
      cudaMalloc(&d_odata, NRUNS * sizeof(int));

      cudaMemcpy(d_idata, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

      cudaMemset(d_odata, 0, NRUNS * sizeof(int));
      // Initializing scratch so as to test racechecking (otherwise we might
      // get errors about accessing uninitialized memory).
      //cudaMemset(d_scratch, 0, NBLOCKS * NRUNS * sizeof(int));
      
      printf("Launching %d blocks of %d threads each " 
             "to asychronously sum the list above %d times.\n", 
             NBLOCKS, NTHREADS, NRUNS);

      cudaDeviceSynchronize();
      for (i = 0; i != NRUNS; ++i) {
            cudaStreamCreate(&streams[i]);
            sum<<< NBLOCKS, NTHREADS, NTHREADS * sizeof(int), streams[i] >>>
                  (d_idata, &d_scratch[NBLOCKS * i]);
            sum<<< 1, NBLOCKS, NBLOCKS * sizeof(int), streams[i] >>>
                  (&d_scratch[NBLOCKS * i], &d_odata[i]);
      }
      cudaDeviceSynchronize();

      cudaMemcpyAsync(h_data, d_odata, NRUNS * sizeof(int), cudaMemcpyDeviceToHost, streams[0]);

      cudaStreamSynchronize(streams[0]);
      cudaDeviceSynchronize();

      printf("OUTPUT: ");
      for(i = 0; i != NRUNS; ++i) {
            cudaStreamDestroy(streams[i]);
            printf(" %d ", h_data[i]);
      }
      printf("\n");

      cudaFree(d_idata);
      cudaFree(d_odata);
      cudaFree(d_scratch);
}
