// From: acc6.its.brooklyn.cuny.edu/~cisc7342/codes/multiplestreams.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define NELEMENTS 16
#define NTHREADS_PER_BLOCK 8
#define NRUNS 2
#define NSTREAMS 3

__global__ void init_array(int *g_data, int factor) { 
      int idx = blockIdx.x * blockDim.x + threadIdx.x;

      g_data[idx] = factor;
}

int correct_data(int *a, const int n, const int c) {
      for(int i = 0; i < n; i++)
            if(a[i] != c)
                  return 0;

      return 1;
}

int main(int argc, char *argv[]) {

      int nstreams = NSTREAMS;               // number of streams for CUDA calls
      int nreps = NRUNS;                 // number of times each experiment is repeated
      int n = NELEMENTS;       // number of ints in the data set
      int nbytes = n * sizeof(int);   // number of data bytes
      dim3 threads, blocks;           // kernel launch configuration
      float elapsed_time, time_memcpy, time_kernel;   // timing variables

      // check the compute capability of the device
      int num_devices=0;

      cudaGetDeviceCount(&num_devices);
      if(0==num_devices) {
            printf("your system does not have a CUDA capable device\n");
            return 1;
      }

      // cudaDeviceProp device_properties;
      // cudaGetDeviceProperties(&device_properties, 0 );
      // if( (1 == device_properties.major) && (device_properties.minor < 1))
      //       printf("%s does not have compute capability 1.1 or later\n\n", device_properties.name);

      // allocate host memory
      int c = 5;                      // value to which the array will be initialized
      int *a = 0;                     // pointer to the array data in host memory

      // allocate host memory (pinned is required for achieve asynchronicity)
      cudaMallocHost((void**)&a, nbytes); 

      // allocate device memory
      int *d_a = 0;             // pointers to data and init value in the device memory
      cudaMalloc((void**)&d_a, nbytes );

      // allocate and initialize an array of stream handles
      cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));

      for(int i = 0; i < nstreams; i++)
            cudaStreamCreate(&(streams[i]) );

      // create CUDA event handles
      cudaEvent_t start_event, stop_event;
      cudaEventCreate(&start_event );
      cudaEventCreate(&stop_event );

      // time memcopy from device
      cudaEventRecord(start_event, 0);  // record in stream-0, to ensure that all previous CUDA calls have completed
      cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, streams[0]);
      cudaEventRecord(stop_event, 0);
      cudaEventSynchronize(stop_event);   // block until the event is actually recorded

      cudaEventElapsedTime(&time_memcpy, start_event, stop_event );
      printf("memcopy: %f\n", time_memcpy);

      // time kernel
      threads.x = NTHREADS_PER_BLOCK;
      threads.y = 1;
      threads.z = 1;

      blocks.x = n/threads.x;
      blocks.y = 1;
      blocks.z = 1;
      cudaEventRecord(start_event, 0);

      init_array<<<blocks, threads, 0, streams[0]>>>(d_a, c);

      cudaEventRecord(stop_event, 0);
      cudaEventSynchronize(stop_event);
      cudaEventElapsedTime(&time_kernel, start_event, stop_event );
      printf("kernel: %f\n", time_kernel);

      //////////////////////////////////////////////////////////////////////
      // time non-streamed execution for reference
      cudaEventRecord(start_event, 0);

      for(int k = 0; k < nreps; k++) {
            init_array<<<blocks, threads>>>(d_a, c);
            cudaMemcpy(a, d_a, nbytes, cudaMemcpyDeviceToHost);
      }

      cudaEventRecord(stop_event, 0);
      cudaEventSynchronize(stop_event);

      cudaEventElapsedTime(&elapsed_time, start_event, stop_event );
      printf("non-streamed: %f (%f expected)\n", elapsed_time / nreps, time_kernel + time_memcpy);

      //////////////////////////////////////////////////////////////////////

      // time execution with nstreams streams
      blocks.x = n / (nstreams*threads.x);
      memset(a, 255, nbytes);     // set host memory bits to all 1s, for testing correctness
      cudaMemset(d_a, 0, nbytes); // set device memory to all 0s, for testing correctness
      cudaEventRecord(start_event, 0);

      for(int k = 0; k < nreps; k++) {
            // asynchronously launch nstreams kernels, each operating on its own portion of data
            for(int i = 0; i < nstreams; i++)
                  init_array<<<blocks, threads, 0, streams[i]>>>(d_a + i * n / nstreams, c);

            // asynchronoously launch nstreams memcopies.  Note that memcopy in stream x will only
            //   commence executing when all previous CUDA calls in stream x have completed
            for(int i = 0; i < nstreams; i++)
                  cudaMemcpyAsync(a + i * n / nstreams, d_a + i * n / nstreams, nbytes / nstreams, cudaMemcpyDeviceToHost, streams[i]);
      }

      cudaEventRecord(stop_event, 0);
      cudaEventSynchronize(stop_event);
      cudaEventElapsedTime(&elapsed_time, start_event, stop_event );
      printf("%d streams: %f (%f expected with compute capability 1.1 or later)\n", nstreams, elapsed_time / nreps, time_kernel + time_memcpy / nstreams);

      // check whether the output is correct
      printf("-------------------------------\n");
      if(correct_data(a, n, c))
            printf("Test PASSED\n");
      else
            printf("Test FAILED\n");

      // release resources
      for(int i = 0; i < nstreams; i++)
            cudaStreamDestroy(streams[i]);
      cudaEventDestroy(start_event);
      cudaEventDestroy(stop_event);
      cudaFreeHost(a);
      cudaFree(d_a);

      return 0;
}

