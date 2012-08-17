load dynamic-c-semantics
load cuda-headers

module CUDA-DEVICE-PROPERTIES is
      including DYNAMIC-C-SEMANTICS
      including CUDA-DEVICE-PROPERTIES-HEADER

      /* Constants. */

      macro cuda-func-cache-prefer-none              = 0
      macro cuda-func-cache-prefer-shared            = 1
      macro cuda-func-cache-prefer-l1                = 2

      macro cuda-limit-stack-size                    = 0
      macro cuda-limit-print-fifo-size               = 1
      macro cuda-limit-malloc-heap-size              = 2

      /* Device properties. */

      macro cuda-cfg-deviceCount                     = 1
      macro cuda-cfg-canAccessPeer                   = 0

      macro cuda-cfg-deviceName                      = "GeForce GT 430"
      macro cuda-cfg-major                           = 2
      macro cuda-cfg-minor                           = 1

      macro cuda-cfg-totalGlobalMem                  = 1072889856
      macro cuda-cfg-sharedMemPerBlock               = 49152
      macro cuda-cfg-regsPerBlock                    = 32768
      macro cuda-cfg-warpSize                        = 32
      macro cuda-cfg-memPitch                        = 2147483647
      macro cuda-cfg-maxThreadsPerBlock              = 1024
      macro cuda-cfg-maxThreadsDimX                  = 1024
      macro cuda-cfg-maxThreadsDimY                  = 1024
      macro cuda-cfg-maxThreadsDimZ                  = 64
      macro cuda-cfg-maxGridSizeX                    = 65535
      macro cuda-cfg-maxGridSizeY                    = 65535
      macro cuda-cfg-maxGridSizeZ                    = 65535
      macro cuda-cfg-clockRate                       = 1460000
      macro cuda-cfg-totalConstMem                   = 65536
      macro cuda-cfg-textureAlignment                = 512
      macro cuda-cfg-deviceOverlap                   = 1
      macro cuda-cfg-multiProcessorCount             = 2
      macro cuda-cfg-kernelExecTimeoutEnabled        = 1
      macro cuda-cfg-integrated                      = 0
      macro cuda-cfg-canMapHostMemory                = 1
      macro cuda-cfg-computeMode                     = 0
      macro cuda-cfg-maxTexture1D                    = 65536
      macro cuda-cfg-maxTexture2DX                   = 65536
      macro cuda-cfg-maxTexture2DY                   = 65535
      macro cuda-cfg-maxTexture3DX                   = 2048
      macro cuda-cfg-maxTexture3DY                   = 2048
      macro cuda-cfg-maxTexture3DZ                   = 2048
      macro cuda-cfg-maxTexture1DLayeredX            = 16384
      macro cuda-cfg-maxTexture1DLayeredY            = 2048
      macro cuda-cfg-maxTexture2DLayeredX            = 16384
      macro cuda-cfg-maxTexture2DLayeredY            = 16384
      macro cuda-cfg-maxTexture2DLayeredZ            = 2048
      macro cuda-cfg-surfaceAlignment                = 512
      macro cuda-cfg-concurrentKernels               = 1
      macro cuda-cfg-ECCEnabled                      = 0
      macro cuda-cfg-pciBusID                        = 1
      macro cuda-cfg-pciDeviceID                     = 0
      macro cuda-cfg-pciDomainID                     = 0
      macro cuda-cfg-tccDriver                       = 0
      macro cuda-cfg-asyncEngineCount                = 1
      macro cuda-cfg-unifiedAddressing               = 1
      macro cuda-cfg-memoryClockRate                 = 900000
      macro cuda-cfg-memoryBusWidth                  = 128
      macro cuda-cfg-l2CacheSize                     = 131072
      macro cuda-cfg-maxThreadsPerMultiProcessor     = 1536
end module
