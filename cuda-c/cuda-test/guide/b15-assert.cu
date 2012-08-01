// From Appendix B.15 of the CUDA-C Programming Guide.

#include <assert.h>

// assert() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef assert
#define assert(arg)
#endif

__global__ void testAssert(void)
{
      int is_one = 1;
      int should_be_one = 0;
      // This will have no effect
      assert(is_one);
      // This will halt kernel execution
      assert(should_be_one);
}
int main(int argc, char* argv[])
{
      testAssert<<<1,1>>>();
      cudaDeviceSynchronize();
      cudaDeviceReset();
      return 0;
}

