load common-c-syntax

kmod CUDA-SYNTAX is
	including #BOOL
	including #EXT-BOOL
	including #NAT
	including #INT
	including #RAT
	including #FLOAT
	including #STRING
	including #CONVERSION
	including #RANDOM
	including #ID
	including K
      including COMMON-C-SYNTAX 
	
      sort CudaSpecifier
      subsort CudaSpecifier < C

/* (from cabs.ml)
and cudaspec =
      CUDA_DEVICE | CUDA_GLOBAL | CUDA_HOST
*/
      op CudaDevice : -> CudaSpecifier
      op CudaGlobal : -> CudaSpecifier
      op CudaHost : -> CudaSpecifier
endkm
