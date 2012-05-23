#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

#define NUM_ELEMENTS 12

// Fake shared memory.
int shared[NUM_ELEMENTS];

__global__ void sum_kernel(int* g_odata, int* g_idata, int n, int run) {
      int i, tid = threadIdx;

      shared[tid] = g_idata[tid];
      __syncthreads();

      if (tid < n/2) {
            shared[tid] = shared[tid] + shared[n/2 + tid];
      }

      __syncthreads();

      if (tid == 0) {
            for (i = 1; i != n/2; ++i) 
                  shared[0] += shared[i];
            g_odata[run] = shared[0];
      }
}

__host__ int main(int argc, char** argv) {
      int* d_idata, *d_odata, *h_data;

      int i, num_elements = 12;
      int nblocks = 1;
      int nthreads = num_elements;
      // nruns should be < num_elements.
      int nruns = 2;


      h_data = malloc(num_elements * sizeof(int));

      printf("INPUT: ");
      for(i = 0; i != num_elements; ++i) {

            h_data[i] = (11 + i * i) % 7;
            printf(" %d ", h_data[i]);
      }
      printf("\n");

      cudaMalloc(&d_idata, num_elements * sizeof(int));
      cudaMalloc(&d_odata, nruns * sizeof(int));

      cudaMemcpy(d_idata, h_data, num_elements * sizeof(int), cudaMemcpyHostToDevice);

      printf("Running sum of %d elements\n", num_elements);

      for (i = 0; i != nruns; ++i) {
            sum_kernel<<< nblocks, nthreads, 2 * num_elements, i >>>
                  (d_odata, d_idata, num_elements, i);
      }
      cudaDeviceSynchronize();

      cudaMemcpyAsync(h_data, d_odata, nruns * sizeof(int), cudaMemcpyDeviceToHost, 1);

      cudaStreamSynchronize(1);

      printf("OUTPUT: ");
      for(i = 0; i != nruns; ++i) {
            printf(" %d ", h_data[i]);
      }
      printf("\n");

      free(h_data);
      cudaFree(d_idata);
      cudaFree(d_odata);
}
