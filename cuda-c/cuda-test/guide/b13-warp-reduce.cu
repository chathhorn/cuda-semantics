// From Appendix B.13 of the CUDA-C Programming Guide.

__global__ void warpReduce() {
      // Seed starting value as inverse lane ID
      int value = 31 – laneId;
      // Use XOR mode to perform butterfly reduction
      for (int i=16; i>=1; i/=2)
            value += __shfl_xor(value, i, 32);
      // “value” now contains the sum across all threads
      printf(“Thread %d final value = %d\n”, threadIdx.x, value);
}

void main() {
      warpReduce<<< 1, 32 >>>();
}

