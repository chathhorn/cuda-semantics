#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

#define NELEMENTS 4
// NRUNS should be < NELEMENTS.
#define NRUNS 2
#define NBLOCKS 2
// NTHREADS_PER_BLOCK*NBLOCKS should equal NELEMENTS
#define NTHREADS_PER_BLOCK 2
// Also, NBLOCKS should equal NTHREADS_PER_BLOCK.

// Sums an array.
__global__ void sum_kernel(int* g_idata, int* g_odata) {
      extern __shared__ int shared[];
      int i, gtid = blockIdx.x * blockDim.x + threadIdx.x;
      int tid = threadIdx.x;

      shared[tid] = g_idata[gtid];
      
      __syncthreads();

      if (tid < NTHREADS_PER_BLOCK/2) {
            shared[tid] += shared[NTHREADS_PER_BLOCK/2 + tid];
      }

      __syncthreads();

      if (tid == 0) {
            for (i = 1; i != NTHREADS_PER_BLOCK/2; ++i) {
                  shared[0] += shared[i];
            }

            g_odata[blockIdx.x] = shared[0];
      }
}

int main(int argc, char** argv) {
      int* d_idata, *d_odata, *d_scratch;
      int i;
      int h_data[NELEMENTS];

      // Use a different stream for every run. 
      cudaStream_t streams[NRUNS];

      printf("INPUT: ");
      for(i = 0; i != NELEMENTS; ++i) {
            h_data[i] = (11 + i * i) % 7;
            printf(" %d ", h_data[i]);
      }
      printf("\n");

      printf("Mallocing scratch.\n");
      cudaMalloc(&d_scratch, NBLOCKS * NRUNS * sizeof(int));
      printf("Mallocing idata.\n");
      cudaMalloc(&d_idata, NELEMENTS * sizeof(int));
      printf("Mallocing odata.\n");
      cudaMalloc(&d_odata, NRUNS * sizeof(int));

      printf("Memcpy idata.\n");
      cudaMemcpy(d_idata, h_data, NELEMENTS * sizeof(int), cudaMemcpyHostToDevice);

      printf("Memset odata.\n");
      cudaMemset(d_odata, 0, NRUNS * sizeof(int));
      // Initializing scratch so as to test racechecking (otherwise we might
      // get errors about accessing uninitialized memory).
      printf("Memset scratch.\n");
      cudaMemset(d_scratch, 0, NBLOCKS * NRUNS * sizeof(int));
      
      printf("Launching %d blocks of %d threads each " 
             "to asychronously sum the list above %d times.\n", 
             NBLOCKS, NTHREADS_PER_BLOCK, NRUNS);

      cudaDeviceSynchronize();
      for (i = 0; i != NRUNS; ++i) {
            cudaStreamCreate(&streams[i]);
            sum_kernel<<< NBLOCKS, NTHREADS_PER_BLOCK, NELEMENTS/NBLOCKS * sizeof(int), streams[i] >>>
                  (d_idata, &d_scratch[NBLOCKS * i]);
            sum_kernel<<< 1, NTHREADS_PER_BLOCK, NBLOCKS * sizeof(int), streams[i] >>>
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
