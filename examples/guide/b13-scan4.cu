// From Appendix B.13 of the CUDA-C Programming Guide.

__global__ void scan4() {
      // Seed sample starting value (inverse of lane ID)
      int value = 31 – laneId;
      // Loop to accumulate scan within my partition.
      // Scan requires log2(n) == 3 steps for 8 threads
      // It works by an accumulated sum up the warp
      // by 1, 2, 4, 8 etc. steps.
      for (int i=1; i<=4; i*=2) {
            // Note: shfl requires all threads being
            // accessed to be active. Therefore we do
            // the __shfl unconditionally so that we
            // can read even from threads which won‟t do a
            // sum, and then conditionally assign the result.
            int n = __shfl_up(value, i, 8);
            if (laneId >= i)
                  value += n;
      }
      printf("Thread %d final value = %d\n", threadIdx.x, value);
}

void main() {
      scan4<<< 1, 32 >>>();
}
