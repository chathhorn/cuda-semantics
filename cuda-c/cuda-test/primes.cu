// From: http://www.mersenneforum.org/showthread.php?t=11900
#include <stdio.h>
//#include <cutil_inline.h>

#define CUDA_SAFE_CALL(x) x

typedef unsigned char u8;

int maxT = 0;

__global__ static void Sieve(u8 * sieve,int * primes, int maxp, int sieve_size)
{ 
      int idx = blockIdx.x * blockDim.x + threadIdx.x; 
      if ((idx>0) && (idx < maxp))
      {
            int ithPrime = primes[idx];
            for(int i=(3*ithPrime)>>1 ;i < sieve_size; i+=ithPrime)
                  // i = (ithPrime-1)/2 + ithPrime though the compiler knew this
                  sieve[i] = 1;
      }
}

__global__ static void Individual(u8 * sieve, int nloc)
{ 
      int idx = blockIdx.x * blockDim.x + threadIdx.x; 
      if (idx < nloc)
      {
            /*  int myprime, delta;
                if (idx == 0) {myprime=2;delta=2;}
                if (idx == 1) {myprime=3;}
                if (idx > 2) {myprime = 6*(idx/2) + 2*(idx%2) - 1;}
                if (idx!=0) delta=2*myprime;
                for (int i=myprime*myprime; i<sieve_size; i+=delta)
                sieve[i] = 1;*/

            // check individual primes by trial division

            sieve[idx] = 0;
            int b = 2;
            if (idx==0 || idx==1) sieve[idx] = 1;
            if (idx==2 || idx==3) sieve[idx] = 0;
            while (b*b <= idx)
            {
                  if (idx%b == 0)
                  {sieve[idx] = 1; break;}
                  b++;
            }
      }
}

bool InitCUDA(void)
{
      int count = 0;
      int i = 0;

      cudaGetDeviceCount(&count);
      if(count == 0) {
            fprintf(stderr, "There is no device.\n");
            return false;
      }

      for(i = 0; i < count; i++) {
            cudaDeviceProp prop;
            if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                  if(prop.major >= 1) {
                        printf("Device %d supports CUDA %d.%d\n",i, prop.major, prop.minor);
                        printf("It has warp size %d, %d regs per block, %d threads per block\n",prop.warpSize, prop.regsPerBlock, prop.maxThreadsPerBlock);
                        printf("max Threads %d x %d x %d\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
                        maxT = prop.maxThreadsDim[0];
                        printf("max Grid %d x %d x %d\n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
                        printf("total constant memory %d\n",prop.totalConstMem);
                        break;
                  }
            }
      }
      if(i == count) {
            fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
            return false;
      }
      cudaSetDevice(i);
      return true;
}

int main(int argc, char** argv)
{
      const int N = 100000000;
      int Nsmall = sqrt(N);

      u8 *device_small_sieve, *device_big_sieve;

      if(!InitCUDA()) {
            return 0;
      } 
      CUDA_SAFE_CALL(cudaMalloc((void**) &device_small_sieve, sizeof(u8) * Nsmall));
      CUDA_SAFE_CALL(cudaMemset(device_small_sieve, 0, sizeof(u8)*Nsmall));

      unsigned int shandle;
      cutCreateTimer(&shandle);
      cutStartTimer(shandle);
      Individual<<<maxT,(Nsmall+maxT-1)/maxT, 0>>> (device_small_sieve,Nsmall);
      CUDA_SAFE_CALL(cudaThreadSynchronize());  
      cutStopTimer(shandle);
      printf("%f milliseconds for small primes\n",cutGetTimerValue(shandle));

      u8* host_copy_of_smallsieve = (u8*)malloc(Nsmall*sizeof(u8));
      printf("%p\n",host_copy_of_smallsieve); fflush(stdout);
      CUDA_SAFE_CALL(cudaMemcpy(host_copy_of_smallsieve,
                        device_small_sieve,
                        sizeof(u8) * Nsmall,
                        cudaMemcpyDeviceToHost));
      printf("%p\n",host_copy_of_smallsieve); fflush(stdout);

      // OK.  We've got an array with 0 at the small primes

      int np = 0, test = 0;
      for (int k=0; k<Nsmall; k++) {test |= host_copy_of_smallsieve[k]; np += 1-host_copy_of_smallsieve[k];}
      if (test != 1) {printf("Something impossible just happened\n\n"); exit(1);}
      printf("%d small primes (< %d) found\n", np, Nsmall);
      int* primes = (int*)malloc(np*sizeof(int));
      int pptr = 0;
      for (int k=0; k<Nsmall; k++) 
            if (host_copy_of_smallsieve[k] == 0)
                  primes[pptr++]=k;

      // except that we needed the primes on the device

      int* device_primes;
      CUDA_SAFE_CALL(cudaMalloc((void**) &device_primes, sizeof(int) * np));
      CUDA_SAFE_CALL(cudaMemcpy(device_primes, primes, sizeof(int)*np, cudaMemcpyHostToDevice));

      // now, set up the big array on the device

      CUDA_SAFE_CALL(cudaMalloc((void**) &device_big_sieve, sizeof(u8) * N/2));
      CUDA_SAFE_CALL(cudaMemset(device_big_sieve, 0, sizeof(u8)*N/2));

      unsigned int thandle; 
      cutCreateTimer(&thandle);
      cutStartTimer(thandle);
      Sieve<<<maxT, (np+maxT-1)/maxT, 0>>>(device_big_sieve, device_primes, np, N/2);
      cudaThreadSynchronize();  
      cutStopTimer(thandle);
      printf("%f milliseconds for big sieve\n",cutGetTimerValue(thandle)); 

      u8* host_sieve = (u8*)malloc(N/2*sizeof(u8));
      cudaMemcpy(host_sieve, device_big_sieve, sizeof(u8) * N/2, cudaMemcpyDeviceToHost);

      cudaFree(device_big_sieve);
      cudaFree(device_small_sieve);
      cudaFree(device_primes);

      int nbig = 0;
      for(int i=0;i < N/2;++i) 
            if (host_sieve[i] == 0) {nbig++;} //printf("%d\n",(i==0)?2:2*i+1);}
            printf("%d big primes (< %d) found\n",nbig,N);

            return 0;
            }
