#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

#define NELEMENTS 12
// NRUNS should be < NELEMENTS.
#define NRUNS 2
#define NBLOCKS 2
// NTHREADS_PER_BLOCK*NBLOCKS should equal NELEMENTS
#define NTHREADS_PER_BLOCK 6

__global__ void sum_kernel(int* g_odata, int* g_idata, int run) {
      __shared__ int shared[NTHREADS_PER_BLOCK];
      int i, gtid = (blockIdx * blockDim) + threadIdx;
      int tid = threadIdx;

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

            g_odata[run] += shared[0];
      }
}

__host__ int main(int argc, char** argv) {
      int* d_idata, *d_odata, *h_data;
      int i;

      // Use a different stream for every run.
      cudaStream_t streams[NRUNS];

      h_data = malloc(NELEMENTS * sizeof(int));

      printf("INPUT: ");
      for(i = 0; i != NELEMENTS; ++i) {

            h_data[i] = (11 + i * i) % 7;
            printf(" %d ", h_data[i]);
      }
      printf("\n");

      cudaMalloc(&d_idata, NELEMENTS * sizeof(int));
      cudaMalloc(&d_odata, NRUNS * sizeof(int));

      cudaMemcpy(d_idata, h_data, NELEMENTS * sizeof(int), cudaMemcpyHostToDevice);

      // TODO: cudaMemset()?
      // Zero d_odata.
      for (i = 0; i != NRUNS; ++i) {
            h_data[i] = 0;
      }
      cudaMemcpy(d_odata, h_data, NRUNS * sizeof(int), cudaMemcpyHostToDevice);
      
      printf("Launching %d blocks of %d threads each " 
             "to asychronously sum the list above %d times.\n", 
             NBLOCKS, NTHREADS_PER_BLOCK, NRUNS);

      for (i = 0; i != NRUNS; ++i) {
            cudaStreamCreate(&streams[i]);
            sum_kernel<<< NBLOCKS, NTHREADS_PER_BLOCK, NELEMENTS * sizeof(int), streams[i] >>>
                  (d_odata, d_idata, i);
      }
      cudaDeviceSynchronize();

      cudaMemcpyAsync(h_data, d_odata, NRUNS * sizeof(int), cudaMemcpyDeviceToHost, streams[0]);

      cudaStreamSynchronize(streams[0]);

      printf("OUTPUT: ");
      for(i = 0; i != NRUNS; ++i) {
            cudaStreamDestroy(streams[i]);
            printf(" %d ", h_data[i]);
      }
      printf("\n");

      free(h_data);
      cudaFree(d_idata);
      cudaFree(d_odata);
}
