#include <stdlib.h> 
#include <stdio.h> 
#include <cuda.h> 

#define N 10

__global__ void square_array(int* a) { 
      int idx = blockIdx.x * blockDim.x + threadIdx.x; 
      if (idx < N) a[idx] = a[idx] * a[idx]; 
}

int main(void) { 
      int host[N];
      int* device;
      int i;
      dim3 grid, block;

      size_t nbytes = N * sizeof(int); 

      grid.x = 1;
      grid.y = 1;
      grid.z = 1;

      block.x = 1;
      block.y = 1;
      block.z = 1;

      int nthreads = 4; 
      int nblocks = N/nthreads + !!(N % nthreads);

      grid.x = nblocks;
      block.x = nthreads;

      cudaMalloc(&device, nbytes);

      for (i = 0; i != N; ++i) 
            host[i] = (int)i;

      cudaMemcpy(device, host, nbytes, cudaMemcpyHostToDevice); 

      square_array<<<grid, block>>>(device); 

      cudaMemcpy(host, device, nbytes, cudaMemcpyDeviceToHost); 

      cudaFree(device); 

      for (i = 0; i != N; ++i) {
            printf("%d: %d\n", i, host[i]);
      }
}

