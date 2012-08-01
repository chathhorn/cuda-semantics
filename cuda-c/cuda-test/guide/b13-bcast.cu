// From Appendix B.13 of the CUDA-C Programming Guide.

__global__ void bcast(int arg) {
      int value;
      if (laneId == 0)
            // Note unused variable for
            value = arg;

      // all threads except lane 0
      value = __shfl(value, 0);

      // Get “value” from lane 0
      if (value != arg)
            printf("Thread %d failed.\n", threadIdx.x);
}

void main() {
      bcast<<< 1, 32 >>>(1234);
}

