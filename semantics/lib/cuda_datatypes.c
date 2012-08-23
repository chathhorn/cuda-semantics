#include <cuda_datatypes.h>

struct cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) {
      struct cudaExtent s;
      s.width = w;
      s.height = h;
      s.depth = d;
      return s;
}

struct cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) {
      struct cudaPitchedPtr s;
      s.ptr = d;
      s.pitch = p;
      s.xsize = xsz;
      s.ysize = ysz;
      return s;
}

struct cudaPos make_cudaPos(size_t x, size_t y, size_t z) {
      struct cudaPos s;
      s.x = x;
      s.y = y;
      s.z = z;
      return s;
}

#define DEF_VEC_MAKE1(T) \
      T ## 1 make_ ## T ## 1 (T x) { \
            T ## 2 v; \
            v.x = x; \
            v.y = y; \
            return v; \
      }
#define DEF_VEC_MAKE2(T) \
      T ## 2 make_ ## T ## 2 (T x, T y) { \
            T ## 2 v; \
            v.x = x; \
            v.y = y; \
            return v; \
      }
#define DEF_VEC_MAKE3(T) \
      T ## 3 make_ ## T ## 3 (T x, T y, T z) { \
            T ## 2 v; \
            v.x = x; \
            v.y = y; \
            v.z = z; \
            return v; \
      }
#define DEF_VEC_MAKE4(T) \
      T ## 4 make_ ## T ## 4 (T x, T y, T z, T w) { \
            T ## 2 v; \
            v.x = x; \
            v.y = y; \
            v.z = z; \
            v.w = w; \
            return v; \
      }

DEF_VEC_MAKE1(char) DEF_VEC_MAKE1(uchar)
DEF_VEC_MAKE2(char) DEF_VEC_MAKE2(uchar)
DEF_VEC_MAKE3(char) DEF_VEC_MAKE3(uchar)
DEF_VEC_MAKE4(char) DEF_VEC_MAKE4(uchar)

DEF_VEC_MAKE1(short) DEF_VEC_MAKE1(ushort)
DEF_VEC_MAKE2(short) DEF_VEC_MAKE2(ushort)
DEF_VEC_MAKE3(short) DEF_VEC_MAKE3(ushort)
DEF_VEC_MAKE4(short) DEF_VEC_MAKE4(ushort)

DEF_VEC_MAKE1(int) DEF_VEC_MAKE1(uint)
DEF_VEC_MAKE2(int) DEF_VEC_MAKE2(uint)
DEF_VEC_MAKE3(int) DEF_VEC_MAKE3(uint)
DEF_VEC_MAKE4(int) DEF_VEC_MAKE4(uint)

DEF_VEC_MAKE1(long) DEF_VEC_MAKE1(ulong)
DEF_VEC_MAKE2(long) DEF_VEC_MAKE2(ulong)
DEF_VEC_MAKE3(long) DEF_VEC_MAKE3(ulong)
DEF_VEC_MAKE4(long) DEF_VEC_MAKE4(ulong)

DEF_VEC_MAKE1(longlong) DEF_VEC_MAKE1(ulonglong)
DEF_VEC_MAKE2(longlong) DEF_VEC_MAKE2(ulonglong)

DEF_VEC_MAKE1(float)
DEF_VEC_MAKE2(float)
DEF_VEC_MAKE3(float)
DEF_VEC_MAKE4(float)

DEF_VEC_MAKE1(double)
DEF_VEC_MAKE2(double)

