load common-c-typing

module CUDA-DECLARATIONS-HEADER is
      syntax K ::= "cuda-check-isglobal" "(" K ")" 
      syntax K ::= "cuda-check-isdevice" "(" Nat ")" 
      syntax K ::= "cuda-check-ishost" "(" Nat ")" 
      syntax K ::= "cuda-decl-function" "(" Id "," K ")"
end module

module CUDA-TYPING-HEADER is
      including COMMON-C-TYPING

      declare isCudaSharedType : Type -> Bool
      declare isCudaHostType : Type -> Bool
      declare isCudaDeviceType : Type -> Bool
      declare isCudaGlobalType : Type -> Bool
end module

module CUDA-MEMORY-HEADER is
      syntax K ::= "cuda-set-props" "(" Nat "," K ")"
      syntax K ::= "cuda-set-mdevice" "(" Nat ")"
      // gid, bid
      syntax K ::= "cuda-shared-loc" "(" Nat "," Nat "," Id ")"
      syntax K ::= "cuda-alloc-shared" "(" Id "," Nat ")"
      syntax K ::= "cuda-free-shared"

      syntax BagItem ::= "cuda-mdevice" // Visible to the device (host or device memory).
                       | "cuda-mshared" // Shared device memory.
                       | "cuda-mhost"   // Visible to the host.
                       | "cuda-mmapped" // Host memory mapped into the device address space.

      syntax SetItem ::= "cuda-access-device-read"
                       | "cuda-access-device-write"
                       | "cuda-access-host-read"
                       | "cuda-access-host-write"

	syntax K ::= "cuda-read-check" "(" K ")"
	syntax K ::= "cuda-write-check" "(" K ")"

      syntax K ::= "cuda-std-malloc-check" "(" Nat ")"

end module

module CUDA-STREAMS-HEADER is
      syntax K ::= "cuda-put-in-stream" "(" K "," Nat ")" 
                 | "cuda-join" "(" Nat ")"
end module

module CUDA-THREADS-HEADER is
      syntax K ::= "cuda-launch-thread" "(" K "," Nat "," Nat "," Nat "," Set ")"
                 | "cuda-grid" "(" K "," Nat "," Nat "," Nat ")"
end module

module CUDA-DEVICES-HEADER is
      syntax K ::= "cuda-device-synchronize"
end module

module CUDA-RACECHECK-HEADER is
      ops   cuda-rc-read-m
            cuda-rc-write-m
            : Nat -> K
            
      syntax K ::= "cuda-rc-read" "(" Nat ")"
                 | "cuda-rc-write" "(" Nat ")"
      syntax K ::= "cuda-rc-sync-shared"
                 | "cuda-rc-sync-global"

      syntax K ::= "cuda-rc-on-delete" "(" Nat "," Bag "," Map ")"
end module

module CUDA-PROPERTIES-HEADER is
      syntax K ::= "gehhhhhh"

      ops   cuda-func-cache-prefer-none
            cuda-func-cache-prefer-shared
            cuda-func-cache-prefer-l1
            : -> Nat

      ops   cuda-limit-stack-size
            cuda-limit-printf-fifo-size
            cuda-limit-malloc-heap-size
            : -> Nat

      ops   cuda-memory-type-host
            cuda-memory-type-device
            : -> Nat

      ops   cuda-cfg-raceChecking
            : -> Bool

      ops   cuda-cfg-driverVersion
            cuda-cfg-runtimeVersion
            : -> Nat

      ops cuda-cfg-deviceName : -> String

      ops   cuda-cfg-cacheConfig
            cuda-cfg-deviceCount
            cuda-cfg-canAccessPeer
            cuda-cfg-computeCapability
            cuda-cfg-major
            cuda-cfg-minor
            cuda-cfg-totalGlobalMem
            cuda-cfg-sharedMemPerBlock
            cuda-cfg-regsPerBlock
            cuda-cfg-warpSize
            cuda-cfg-memPitch
            cuda-cfg-maxThreadsPerBlock
            cuda-cfg-maxThreadsDimX
            cuda-cfg-maxThreadsDimY
            cuda-cfg-maxThreadsDimZ
            cuda-cfg-maxGridSizeX
            cuda-cfg-maxGridSizeY
            cuda-cfg-maxGridSizeZ
            cuda-cfg-clockRate
            cuda-cfg-totalConstMem                        
            cuda-cfg-textureAlignment                     
            cuda-cfg-deviceOverlap                        
            cuda-cfg-multiProcessorCount                  
            cuda-cfg-kernelExecTimeoutEnabled             
            cuda-cfg-integrated                           
            cuda-cfg-canMapHostMemory                     
            cuda-cfg-computeMode                          
            cuda-cfg-maxTexture1D                         
            cuda-cfg-maxTexture2DX                        
            cuda-cfg-maxTexture2DY                        
            cuda-cfg-maxTexture3DX                        
            cuda-cfg-maxTexture3DY                        
            cuda-cfg-maxTexture3DZ                        
            cuda-cfg-maxTexture1DLayeredX                 
            cuda-cfg-maxTexture1DLayeredY                 
            cuda-cfg-maxTexture2DLayeredX                 
            cuda-cfg-maxTexture2DLayeredY                 
            cuda-cfg-maxTexture2DLayeredZ                 
            cuda-cfg-surfaceAlignment                     
            cuda-cfg-concurrentKernels                    
            cuda-cfg-ECCEnabled                           
            cuda-cfg-pciBusID                             
            cuda-cfg-pciDeviceID                          
            cuda-cfg-pciDomainID                          
            cuda-cfg-tccDriver                            
            cuda-cfg-asyncEngineCount                     
            cuda-cfg-unifiedAddressing                    
            cuda-cfg-memoryClockRate                      
            cuda-cfg-memoryBusWidth                       
            cuda-cfg-l2CacheSize                          
            cuda-cfg-maxThreadsPerMultiProcessor
            : -> Nat
end module

module CUDA-ERRORS-HEADER is
      syntax K ::= "cuda-return-success"
                 | "cuda-return" "(" Nat ")"

      ops   cuda-success                           
            cuda-error-missing-config         
            cuda-error-memory-allocation       
            cuda-error-initialization-error          
            cuda-error-launch-failure                
            cuda-error-prior-launch-failure           
            cuda-error-launch-timeout                
            cuda-error-launch-out-of-resources         
            cuda-error-invalid-device-function        
            cuda-error-invalid-config
            cuda-error-invalid-device                
            cuda-error-invalid-value                 
            cuda-error-invalid-pitch-value            
            cuda-error-invalid-symbol                
            cuda-error-map-buffer-object-failed        
            cuda-error-unmap-buffer-object-failed      
            cuda-error-invalid-host-pointer           
            cuda-error-invalid-device-pointer         
            cuda-error-invalid-texture               
            cuda-error-invalid-texture-binding        
            cuda-error-invalid-channel-descriptor     
            cuda-error-invalid-memcpy-direction       
            cuda-error-address-of-constant            
            cuda-error-texture-fetch-failed           
            cuda-error-texture-not-bound              
            cuda-error-synchronization-error         
            cuda-error-invalid-filter-setting         
            cuda-error-invalid-norm-setting           
            cuda-error-mixed-device-execution         
            cuda-error-cudart-unloading              
            cuda-error-unknown                      
            cuda-error-not-yet-implemented            
            cuda-error-memory-value-too-large          
            cuda-error-invalid-resource-handle        
            cuda-error-not-ready                     
            cuda-error-insufficient-driver           
            cuda-error-set-on-active-process           
            cuda-error-invalid-surface-reference
            cuda-error-no-device                     
            cuda-error-ecc-uncorrectable             
            cuda-error-shared-object-symbol-not-found                                 
            cuda-error-shared-object-init-failed                                      
            cuda-error-unsupported-limit                                              
            cuda-error-duplicate-variable-name                                        
            cuda-error-duplicate-texture-name                                         
            cuda-error-duplicate-surface-name                                         
            cuda-error-devices-unavailable                                            
            cuda-error-invalid-kernel-image                                           
            cuda-error-no-kernel-image-for-device                                     
            cuda-error-incompatible-driver
            cuda-error-peer-access-already-enabled                                    
            cuda-error-peer-access-not-enabled                                        
            cuda-error-device-already-in-use                                          
            cuda-error-profiler-disabled                                              
            cuda-error-profiler-not-initialized                                       
            cuda-error-profiler-already-started                                       
            cuda-error-profiler-already-stopped                                       
            cuda-error-assert                                      
            cuda-error-too-many-peers                              
            cuda-error-host-memory-already-registered              
            cuda-error-host-memory-not-registered                  
            cuda-error-operating-system                            
            cuda-error-startup-failure               
            cuda-error-api-failure-base               
            : -> Nat
end module

module CUDA-DATATYPES-HEADER is
      including COMMON-C-TYPING

      ops cuda-dim3-type : -> Type
end module

