// Incomplete
// From Appendix B.5 of the CUDA-C Programming Guide. 

__device__ unsigned int count = 0;

__shared__ bool isLastBlockDone;

__global__ void sum(const float* array, unsigned int N, float* result) {

      // Each block sums a subset of the input array
      float partialSum = calculatePartialSum(array, N);

      if (threadIdx.x == 0) {
            // Thread 0 of each block stores the partial sum
            // to global memory
            result[blockIdx.x] = partialSum;
            // Thread 0 makes sure its result is visible to
            // all other threads
            __threadfence();
            // Thread 0 of each block signals that it is done
            unsigned int value = atomicInc(&count, gridDim.x);
            // Thread 0 of each block determines if its block is
            // the last block to be done
            isLastBlockDone = (value == (gridDim.x - 1));
      }

      // Synchronize to make sure that each thread reads
      // the correct value of isLastBlockDone
      __syncthreads();
      if (isLastBlockDone) {
            // The last block sums the partial sums
            // stored in result[0 .. gridDim.x-1]

            float totalSum = calculateTotalSum(result);
            if (threadIdx.x == 0) {
                  // Thread 0 of last block stores total sum
                  // to global memory and resets count so that
                  // next kernel call works properly
                  result[0] = totalSum;
                  count = 0;
            }
      }
}


