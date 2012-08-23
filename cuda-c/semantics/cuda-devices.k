load dynamic-c-semantics
load cuda-headers

module CUDA-DEVICES is
      including DYNAMIC-C-SEMANTICS
      including CUDA-DEVICES-HEADER
      including CUDA-RACECHECK-HEADER
      including CUDA-PROPERTIES-HEADER
      including CUDA-DATATYPES-HEADER
      including CUDA-ERRORS-HEADER

      // TODO
      rule [cudaSetDeviceFlags]:
            <k> prepareBuiltin((Identifier("cudaSetDeviceFlags")), _)
                  => cuda-return-success
            ...</k>

      // TODO check args
      rule [cudaSetValidDevices]:
            <k> prepareBuiltin((Identifier("cudaSetValidDevices")), (PtrDevices:KResult,, tv(Len:Nat, _)))
                  => cuda-return-success
            ...</k>

      rule [cudaDeviceGetCacheConfig]:
            <k> prepareBuiltin((Identifier("cudaDeviceGetCacheConfig")), PtrValue:KResult)
                  => Computation(*(PtrValue) := tv(CacheConfig:Nat, t(.Set, int)))
                  ~> cuda-return-success
            ...</k>
            <cudaDeviceCacheConfig> CacheConfig </cudaDeviceCacheConfig>

      rule [cudaDeviceSetCacheConfig]:
            <k> prepareBuiltin((Identifier("cudaDeviceSetCacheConfig")), tv(CacheConfig:Nat, _))
                  => cuda-return-success
            ...</k>
            <cudaDeviceCacheConfig> _ => CacheConfig </cudaDeviceCacheConfig>

      rule [cudaDeviceGetLimit-stack]:
            <k> prepareBuiltin((Identifier("cudaDeviceGetLimit")), (PtrValue:KResult,, tv(LimitKind:Nat, _)))
                  => Computation(*(PtrValue) := tv(Limit:Nat, t(.Set, int)))
                  ~> cuda-return-success
            ...</k>
            <cudaLimitStackSize> Limit </cudaLimitStackSize>
            when LimitKind ==Int cuda-limit-stack-size
      rule [cudaDeviceGetLimit-printfifo]:
            <k> prepareBuiltin((Identifier("cudaDeviceGetLimit")), (PtrValue:KResult,, tv(LimitKind:Nat, _)))
                  => Computation(*(PtrValue) := tv(Limit:Nat, t(.Set, int)))
                  ~> cuda-return-success
            ...</k>
            <cudaLimitPrintFifoSize> Limit </cudaLimitPrintFifoSize>
            when LimitKind ==Int cuda-limit-printf-fifo-size
      rule [cudaDeviceGetLimit-heap]:
            <k> prepareBuiltin((Identifier("cudaDeviceGetLimit")), (PtrValue:KResult,, tv(LimitKind:Nat, _)))
                  => Computation(*(PtrValue) := tv(Limit:Nat, t(.Set, int)))
                  ~> cuda-return-success
            ...</k>
            <cudaLimitMallocHeapSize> Limit </cudaLimitMallocHeapSize>
            when LimitKind ==Int cuda-limit-malloc-heap-size
      rule [cudaDeviceGetLimit-invalid]:
            <k> prepareBuiltin((Identifier("cudaDeviceGetLimit")), (_,, tv(LimitKind:Nat, _)))
                  => cuda-return(cuda-error-unsupported-limit)
            ...</k>
            when LimitKind =/=Int cuda-limit-stack-size
            andBool LimitKind =/=Int cuda-limit-printf-fifo-size
            andBool LimitKind =/=Int cuda-limit-malloc-heap-size
      
      rule [cudaDeviceSetLimit-stack]:
            <k> prepareBuiltin((Identifier("cudaDeviceSetLimit")), (tv(LimitKind:Nat, _),, tv(Limit:Nat, _)))
                  => cuda-return-success
            ...</k>
            <cudaLimitStackSize> _ => Limit </cudaLimitStackSize>
            when LimitKind ==Int cuda-limit-stack-size
      rule [cudaDeviceSetLimit-printfifo]:
            <k> prepareBuiltin((Identifier("cudaDeviceSetLimit")), (tv(LimitKind:Nat, _),, tv(Limit:Nat, _)))
                  => cuda-return-success
            ...</k>
            <cudaLimitPrintFifoSize> _ => Limit </cudaLimitPrintFifoSize>
            when LimitKind ==Int cuda-limit-printf-fifo-size
      rule [cudaDeviceSetLimit-heap]:
            <k> prepareBuiltin((Identifier("cudaDeviceSetLimit")), (tv(LimitKind:Nat, _),, tv(Limit:Nat, _)))
                  => cuda-return-success
            ...</k>
            <cudaLimitMallocHeapSize> _ => Limit </cudaLimitMallocHeapSize>
            when LimitKind ==Int cuda-limit-malloc-heap-size
      rule [cudaDeviceSetLimit-invalid]:
            <k> prepareBuiltin((Identifier("cudaDeviceSetLimit")), (tv(LimitKind:Nat, _),, _))
                  => cuda-return(cuda-error-unsupported-limit)
            ...</k>
            when LimitKind =/=Int cuda-limit-stack-size
            andBool LimitKind =/=Int cuda-limit-printf-fifo-size
            andBool LimitKind =/=Int cuda-limit-malloc-heap-size

      // We've got one device so the choice is pretty easy.
      rule [cudaChooseDevice]:
            <k> prepareBuiltin((Identifier("cudaChooseDevice")), (PtrDevice:KResult,, _))
                  => Computation(*(PtrDevice) := tv(0, t(.Set, int)))
                  ~> cuda-return-success
            ...</k>

      rule [cudaDeviceGetCacheConfig]:
            <k> prepareBuiltin((Identifier("cudaDeviceGetCacheConfig")), PtrCacheConfig:KResult)
                  => Computation(*(PtrCacheConfig) := tv(cuda-cfg-cacheConfig, t(.Set, int)))
                  ~> cuda-return-success
            ...</k>

      rule [cudaGetDevice]:
            <k> prepareBuiltin((Identifier("cudaGetDevice")), PtrDevice:KResult)
                  => Computation(*(PtrDevice) := tv(0, t(.Set, int)))
                  ~> cuda-return-success
            ...</k>

      rule [cudaSetDevice]:
            <k> prepareBuiltin((Identifier("cudaSetDevice")), tv(0, _))
                  ~> cuda-return-success
            ...</k>
      rule [cudaSetDevice-invalid]:
            <k> prepareBuiltin((Identifier("cudaSetDevice")), tv(Device:Nat, _))
                  ~> cuda-return(cuda-error-invalid-device)
            ...</k>
            when Device =/=Int 0

      rule [cudaGetDeviceCount]:
            <k> prepareBuiltin((Identifier("cudaGetDeviceCount")), PtrCount:KResult)
                  => Computation(*(PtrCount) := tv(cuda-cfg-deviceCount, t(.Set, int)))
                  ~> cuda-return-success
            ...</k>

      rule [cudaDeviceCanAccessPeer]:
            <k> prepareBuiltin((Identifier("cudaDeviceCanAccessPeer")), (PtrAccess:KResult,, tv(Device:Nat, _),, tv(PeerDevice:Nat, _)))
                  => Computation(*(PtrAccess) := tv(cuda-cfg-canAccessPeer, t(.Set, int)))
                  ~> cuda-return-success
            ...</k>
            when Device:Nat ==Int 0
            andBool PeerDevice:Nat ==Int 0
      rule [cudaDeviceCanAccessPeer-invalid]:
            <k> prepareBuiltin((Identifier("cudaDeviceCanAccessPeer")), (_,, tv(Device:Nat, _),, tv(PeerDevice:Nat, _)))
                  => cuda-return(cuda-error-invalid-device)
            ...</k>
            when Device:Nat =/=Int 0
            orBool PeerDevice:Nat =/=Int 0

      rule [cudaDeviceDisablePeerAccess]:
            <k> prepareBuiltin((Identifier("cudaDeviceDisablePeerAccess")), tv(PeerDevice:Nat, _))
                  => cuda-return(cuda-error-peer-access-not-enabled)
            ...</k>
            when PeerDevice:Nat ==Int 0
      rule [cudaDeviceDisablePeerAccess-invalid]:
            <k> prepareBuiltin((Identifier("cudaDeviceDisablePeerAccess")), tv(PeerDevice:Nat, _))
                  => cuda-return(cuda-error-invalid-device)
            ...</k>
            when PeerDevice:Nat =/=Int 0

      rule [cudaDeviceEnablePeerAccess]:
            <k> prepareBuiltin((Identifier("cudaDeviceEnablePeerAccess")), _)
                  => cuda-return(cuda-error-invalid-device)
            ...</k>

      rule [cudaDeviceReset]:
            <k> prepareBuiltin((Identifier("cudaDeviceReset")), _)
                  => cuda-return-success
            ...</k>

      rule [cudaGetDeviceProperties-invalid]:
            <k> prepareBuiltin((Identifier("cudaGetDeviceProperties")), (_,, tv(Device:Nat, _)))
                  => cuda-return(cuda-error-invalid-device)
            ...</k>
            when Device:Nat =/=Int 0
      rule [cudaGetDeviceProperties]:
            <k> prepareBuiltin((Identifier("cudaGetDeviceProperties")), (PtrProps:KResult,, tv(0, _)))
                  => writeString((*(PtrProps) . Identifier("name")), cuda-cfg-deviceName +String "\0")
                  ~> Computation(*(PtrProps) . Identifier("major") := tv(cuda-cfg-major, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("minor") := tv(cuda-cfg-minor, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("totalGlobalMem") := tv(cuda-cfg-totalGlobalMem, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("sharedMemPerBlock") := tv(cuda-cfg-sharedMemPerBlock, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("regsPerBlock") := tv(cuda-cfg-regsPerBlock, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("warpSize") := tv(cuda-cfg-warpSize, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("memPitch") := tv(cuda-cfg-memPitch, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxThreadsPerBlock") := tv(cuda-cfg-maxThreadsPerBlock, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxThreadsDim") [ 0 ] := tv(cuda-cfg-maxThreadsDimX, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxThreadsDim") [ 1 ] := tv(cuda-cfg-maxThreadsDimY, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxThreadsDim") [ 2 ] := tv(cuda-cfg-maxThreadsDimZ, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxGridSize") [ 0 ] := tv(cuda-cfg-maxGridSizeX, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxGridSize") [ 1 ] := tv(cuda-cfg-maxGridSizeY, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxGridSize") [ 2 ] := tv(cuda-cfg-maxGridSizeZ, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("clockRate") := tv(cuda-cfg-clockRate, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("totalConstMem") := tv(cuda-cfg-totalConstMem, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("textureAlignment") := tv(cuda-cfg-textureAlignment, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("deviceOverlap") := tv(cuda-cfg-deviceOverlap, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("multiProcessorCount") := tv(cuda-cfg-multiProcessorCount, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("kernelExecTimeoutEnabled") := tv(cuda-cfg-kernelExecTimeoutEnabled, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("integrated") := tv(cuda-cfg-integrated, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("canMapHostMemory") := tv(cuda-cfg-canMapHostMemory, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("computeMode") := tv(cuda-cfg-computeMode, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxTexture1D") := tv(cuda-cfg-maxTexture1D, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxTexture2D") [ 0 ] := tv(cuda-cfg-maxTexture2DX, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxTexture2D") [ 1 ] := tv(cuda-cfg-maxTexture2DY, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxTexture3D") [ 0 ] := tv(cuda-cfg-maxTexture3DX, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxTexture3D") [ 1 ] := tv(cuda-cfg-maxTexture3DY, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxTexture3D") [ 2 ] := tv(cuda-cfg-maxTexture3DZ, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxTexture1DLayered") [ 0 ] := tv(cuda-cfg-maxTexture1DLayeredX, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxTexture1DLayered") [ 1 ] := tv(cuda-cfg-maxTexture1DLayeredY, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxTexture2DLayered") [ 0 ] := tv(cuda-cfg-maxTexture2DLayeredX, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxTexture2DLayered") [ 1 ] := tv(cuda-cfg-maxTexture2DLayeredY, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxTexture2DLayered") [ 2 ] := tv(cuda-cfg-maxTexture2DLayeredZ, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("surfaceAlignment") := tv(cuda-cfg-surfaceAlignment, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("concurrentKernels") := tv(cuda-cfg-concurrentKernels, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("ECCEnabled") := tv(cuda-cfg-ECCEnabled, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("pciBusID") := tv(cuda-cfg-pciBusID, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("pciDeviceID") := tv(cuda-cfg-pciDeviceID, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("pciDomainID") := tv(cuda-cfg-pciDomainID, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("tccDriver") := tv(cuda-cfg-tccDriver, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("asyncEngineCount") := tv(cuda-cfg-asyncEngineCount, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("unifiedAddressing") := tv(cuda-cfg-unifiedAddressing, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("memoryClockRate") := tv(cuda-cfg-memoryClockRate, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("memoryBusWidth") := tv(cuda-cfg-memoryBusWidth, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("l2CacheSize") := tv(cuda-cfg-l2CacheSize, t(.Set, int)))
                  ~> Computation(*(PtrProps) . Identifier("maxThreadsPerMultiProcessor") := tv(cuda-cfg-l2CacheSize, t(.Set, int)))
                  ~> cuda-return-success
            ...</k>

      /*@ Device synchronization. */

      rule [cudaDeviceSynchronize]:
            <k> prepareBuiltin((Identifier("cudaDeviceSynchronize")), _) => cuda-device-synchronize ...</k>
      rule [cuda-device-synchronize]:
           <k> cuda-device-synchronize => cuda-rc-sync-global ~> cuda-return-success ...</k>
           <activeStreams> .Set </activeStreams> // i.e., all streams are empty.

end module
