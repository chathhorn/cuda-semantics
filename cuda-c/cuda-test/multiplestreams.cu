// From: acc6.its.brooklyn.cuny.edu/~cisc7342/codes/multiplestreams.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define N 16
#define NTHREADS_PER_BLOCK 8
#define NRUNS 2
// grid.x = N/NTHREADS_PER_BLOCK on the first run and
// grid.x = N/(NSTREAMS*NTHREADS_PER_BLOCK) on the second.
#define NSTREAMS 2


__global__ void init_array(int* g_data, int factor) { 
      int idx = blockIdx.x * blockDim.x + threadIdx.x;

      g_data[idx] = factor;
}

int correct_data(int* a, int n, int c) {
      int i;

      for(i = 0; i != n; ++i) {
            if (a[i] != c) return 0;
      }

      return 1;
}

int main(int argc, char *argv[]) {

      int nbytes = N * sizeof(int);   // number of data bytes
      dim3 block, grid;           // kernel launch configuration
      float elapsed_time, time_memcpy, time_kernel;   // timing variables
      int i, j;

      // check the compute capability of the device
      int num_devices = 0;
      int c = 5;                      // value to which the array will be initialized
      int* a = 0;                     // pointer to the array data in host memory
      int* d_a = 0;             // pointers to data and init value in the device memory

      cudaGetDeviceCount(&num_devices);
      if (num_devices == 0) {
            printf("Your system does not have a CUDA capable device.\n");
            return 1;
      }

      // cudaDeviceProp device_properties;
      // cudaGetDeviceProperties(&device_properties, 0 );
      // if( (1 == device_properties.major) && (device_properties.minor < 1))
      //       printf("%s does not have compute capability 1.1 or later\n\n", device_properties.name);

      // allocate host memory

      // allocate host memory (pinned is required for achieve asynchronicity)
      cudaMallocHost((void**)&a, nbytes); 

      // allocate device memory
      cudaMalloc((void**)&d_a, nbytes);

      // allocate and initialize an array of stream handles
      cudaStream_t* streams = (cudaStream_t*) malloc(NSTREAMS * sizeof(cudaStream_t));

      for (i = 0; i != NSTREAMS; ++i) cudaStreamCreate(&streams[i]);

      // create CUDA event handles
      cudaEvent_t start_event, stop_event;
      cudaEventCreate(&start_event );
      cudaEventCreate(&stop_event );

      // time memcopy from device
      cudaEventRecord(start_event, 0);  // record in stream-0, to ensure that all previous CUDA calls have completed
      cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, streams[0]);
      cudaEventRecord(stop_event, 0);
      cudaEventSynchronize(stop_event);   // block until the event is actually recorded

      cudaEventElapsedTime(&time_memcpy, start_event, stop_event);
      printf("memcopy: %f\n", time_memcpy);

      cudaEventRecord(start_event, 0);

      grid.x = N/NTHREADS_PER_BLOCK;
      grid.y = 1;
      grid.z = 1;

      block.x = NTHREADS_PER_BLOCK;
      block.y = 1;
      block.z = 1;

      init_array<<<grid, block, 0, streams[0]>>>(d_a, c);

      cudaEventRecord(stop_event, 0);
      cudaEventSynchronize(stop_event);
      cudaEventElapsedTime(&time_kernel, start_event, stop_event );
      printf("kernel: %f\n", time_kernel);

      //////////////////////////////////////////////////////////////////////
      // time non-streamed execution for reference
      cudaEventRecord(start_event, 0);

      for (i = 0; i != NRUNS; ++i) {
            init_array<<<grid, block>>>(d_a, c);
            cudaMemcpy(a, d_a, nbytes, cudaMemcpyDeviceToHost);
      }

      cudaEventRecord(stop_event, 0);
      cudaEventSynchronize(stop_event);

      cudaEventElapsedTime(&elapsed_time, start_event, stop_event );
      printf("non-streamed: %f (%f expected)\n", 
            elapsed_time / NRUNS, time_kernel + time_memcpy);

      //////////////////////////////////////////////////////////////////////

      // time execution with NSTREAMS streams
      grid.x = N / (NSTREAMS*NTHREADS_PER_BLOCK);
      memset(a, 255, nbytes);     // set host memory bits to all 1s, for testing correctness
      cudaMemset(d_a, 0, nbytes); // set device memory to all 0s, for testing correctness
      cudaEventRecord(start_event, 0);

      for (i = 0; i != NRUNS; ++i) {
            // asynchronously launch NSTREAMS kernels, each operating on its own portion of data
            for (j = 0; j != NSTREAMS; ++j) {
                  init_array<<<grid, block, 0, streams[j]>>>(d_a + j * N / NSTREAMS, c);
            }

            // asynchronoously launch NSTREAMS memcopies.  Note that memcopy in stream x will only
            //   commence executing when all previous CUDA calls in stream x have completed
            for (j = 0; j != NSTREAMS; ++j) {
                  cudaMemcpyAsync(a + j * N / NSTREAMS, 
                        d_a + j * N / NSTREAMS, nbytes / NSTREAMS, 
                        cudaMemcpyDeviceToHost, streams[j]);
            }
      }

      cudaEventRecord(stop_event, 0);
      cudaEventSynchronize(stop_event);
      cudaEventElapsedTime(&elapsed_time, start_event, stop_event );
      printf("%d streams: %f (%f expected with compute capability 1.1 or later)\n", 
            NSTREAMS, elapsed_time / NRUNS, time_kernel + time_memcpy / NSTREAMS);

      // check whether the output is correct
      printf("-------------------------------\n");
      if (correct_data(a, N, c)) {
            printf("Test PASSED\n");
      } else {
            printf("Test FAILED\n");
      }

      // release resources
      for (i = 0; i != NSTREAMS; ++i) cudaStreamDestroy(streams[i]);

      cudaEventDestroy(start_event);
      cudaEventDestroy(stop_event);
      cudaFreeHost(a);
      cudaFree(d_a);

      return 0;
}

