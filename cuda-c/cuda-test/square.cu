/* From: http://llpanorama.wordpress.com/2008/05/21/my-first-cuda-program/ */
#include <stdlib.h> 
#include <stdio.h> 
#include <cuda.h> 

#define N 10

__global__ void square_array(int* a) { 
      int idx = blockIdx.x * blockDim.x + threadIdx.x; 
      if (idx < N) a[idx] = a[idx] * a[idx]; 
}

int main(void) { 
      int* hostptr, *devptr;
      int i;

      size_t nbytes = N * sizeof(int); 

      dim3 grid = {1, 1, 1};
      dim3 block = {1, 1, 1};

      int nthreads = 4; 
      int nblocks = N/nthreads + !!(N % nthreads);

      grid.x = nblocks;
      block.x = nthreads;

      hostptr = (int*) malloc(nbytes);
      cudaMalloc(&devptr, nbytes);

      for (i = 0; i != N; ++i) 
            hostptr[i] = (int)i;

      cudaMemcpy(devptr, hostptr, nbytes, cudaMemcpyHostToDevice); 

      square_array<<<grid, block>>>(devptr); 

      cudaMemcpy(hostptr, devptr, nbytes, cudaMemcpyDeviceToHost); 

      for (i = 0; i != N; ++i) 
            printf("%d %d\n", i, hostptr[i]);

      cudaFree(devptr); 
      free(hostptr);
}

