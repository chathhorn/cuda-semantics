
__global__ int gfunc(int a) {
      return a * 8;
}

__device__ int dfunc(int b) {
      return b * b;
}

__host__ int main(int argc, char** argv) {
      gfunc(2);
      dfunc(3);
}
