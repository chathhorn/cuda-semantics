#include <stdlib.h> 
#include <stdio.h> 
#include <cuda.h> 

__global__ void square_array(float *a, int N) { 
      //int idx = blockIdx.x * blockDim.x + threadIdx.x; 
      //if (idx < N) a[idx] = a[idx] * a[idx]; 
      printf("bam");
}

// main routine that executes on the host
int main(void) { 
      float *hostptr, *devptr;
      int N = 10, i;

      size_t nbytes = N * sizeof(float); 

      int nthreads = 4; 
      int nblocks = N/nthreads + !!(N % nthreads);

      hostptr = (float*) malloc(nbytes);
      devptr = (float*) malloc(nbytes);
      //cudaMalloc((void**) &devptr, nbytes);

      for (i = 0; i != N; ++i) 
            hostptr[i] = (float)i;

      //cudaMemcpy(devptr, hostptr, nbytes, cudaMemcpyHostToDevice); 
      square_array<<<nblocks, nthreads>>>(devptr, N); 
      //cudaMemcpy(hostptr, devptr, nbytes, cudaMemcpyDeviceToHost); 

      for (i = 0; i != N; ++i) 
            printf("%d %f\n", i, hostptr[i]);

      //cudaFree(devptr); 
      free(hostptr);
}

