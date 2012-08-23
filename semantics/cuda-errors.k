load dynamic-c-semantics
load cuda-headers

module CUDA-ERRORS is
      including DYNAMIC-C-SEMANTICS
      including CUDA-ERRORS-HEADER

      macro cuda-success                              =     0
      macro cuda-error-missing-config                 =     1
      macro cuda-error-memory-allocation              =     2
      macro cuda-error-initialization-error           =     3
      macro cuda-error-launch-failure                 =     4
      macro cuda-error-prior-launch-failure           =     5
      macro cuda-error-launch-timeout                 =     6
      macro cuda-error-launch-out-of-resources        =     7
      macro cuda-error-invalid-device-function        =     8
      macro cuda-error-invalid-config                 =     9
      macro cuda-error-invalid-device                 =    10
      macro cuda-error-invalid-value                  =    11
      macro cuda-error-invalid-pitch-value            =    12
      macro cuda-error-invalid-symbol                 =    13
      macro cuda-error-map-buffer-object-failed       =    14
      macro cuda-error-unmap-buffer-object-failed     =    15
      macro cuda-error-invalid-host-pointer           =    16
      macro cuda-error-invalid-device-pointer         =    17
      macro cuda-error-invalid-texture                =    18
      macro cuda-error-invalid-texture-binding        =    19
      macro cuda-error-invalid-channel-descriptor     =    20
      macro cuda-error-invalid-memcpy-direction       =    21
      macro cuda-error-address-of-constant            =    22
      macro cuda-error-texture-fetch-failed           =    23
      macro cuda-error-texture-not-bound              =    24
      macro cuda-error-synchronization-error          =    25
      macro cuda-error-invalid-filter-setting         =    26
      macro cuda-error-invalid-norm-setting           =    27
      macro cuda-error-mixed-device-execution         =    28
      macro cuda-error-cudart-unloading               =    29
      macro cuda-error-unknown                        =    30
      macro cuda-error-not-yet-implemented            =    31
      macro cuda-error-memory-value-too-large         =    32
      macro cuda-error-invalid-resource-handle        =    33
      macro cuda-error-not-ready                      =    34
      macro cuda-error-insufficient-driver            =    35
      macro cuda-error-set-on-active-process          =    36
      macro cuda-error-invalid-surface-reference      =    37
      macro cuda-error-no-device                      =    38
      macro cuda-error-ecc-uncorrectable              =    39
      macro cuda-error-shared-object-symbol-not-found =    40
      macro cuda-error-shared-object-init-failed      =    41
      macro cuda-error-unsupported-limit              =    42
      macro cuda-error-duplicate-variable-name        =    43
      macro cuda-error-duplicate-texture-name         =    44
      macro cuda-error-duplicate-surface-name         =    45
      macro cuda-error-devices-unavailable            =    46
      macro cuda-error-invalid-kernel-image           =    47
      macro cuda-error-no-kernel-image-for-device     =    48
      macro cuda-error-incompatible-driver            =    49
      macro cuda-error-peer-access-already-enabled    =    50
      macro cuda-error-peer-access-not-enabled        =    51

      macro cuda-error-device-already-in-use          =    54
      macro cuda-error-profiler-disabled              =    55
      macro cuda-error-profiler-not-initialized       =    56
      macro cuda-error-profiler-already-started       =    57
      macro cuda-error-profiler-already-stopped       =    58
      macro cuda-error-assert                         =    59
      macro cuda-error-too-many-peers                 =    60
      macro cuda-error-host-memory-already-registered =    61
      macro cuda-error-host-memory-not-registered     =    62
      macro cuda-error-operating-system               =    63

      macro cuda-error-startup-failure                =   127
      macro cuda-error-api-failure-base               = 10000

      syntax K ::= "cuda-get-error-string" "(" Nat ")"
      rule cuda-get-error-string(0) => "no error" [structural anywhere]
      rule cuda-get-error-string(1) => "__global__ function call is not configured" [structural anywhere]
      rule cuda-get-error-string(2) => "out of memory" [structural anywhere]
      rule cuda-get-error-string(3) => "initialization error" [structural anywhere]
      rule cuda-get-error-string(4) => "unspecified launch failure" [structural anywhere]
      rule cuda-get-error-string(5) => "unspecified launch failure in prior launch" [structural anywhere]
      rule cuda-get-error-string(6) => "the launch timed out and was terminated" [structural anywhere]
      rule cuda-get-error-string(7) => "too many resources requested for launch" [structural anywhere]
      rule cuda-get-error-string(8) => "invalid device function " [structural anywhere]
      rule cuda-get-error-string(9) => "invalid config argument" // TODO: config for configuration keeps K from spazzing. 
            [structural anywhere]
      rule cuda-get-error-string(10) => "invalid device ordinal" [structural anywhere]
      rule cuda-get-error-string(11) => "invalid argument" [structural anywhere]
      rule cuda-get-error-string(12) => "invalid pitch argument" [structural anywhere]
      rule cuda-get-error-string(13) => "invalid device symbol" [structural anywhere]
      rule cuda-get-error-string(14) => "mapping of buffer object failed" [structural anywhere]
      rule cuda-get-error-string(15) => "unmapping of buffer object failed" [structural anywhere]
      rule cuda-get-error-string(16) => "invalid host pointer" [structural anywhere]
      rule cuda-get-error-string(17) => "invalid device pointer" [structural anywhere]
      rule cuda-get-error-string(18) => "invalid texture reference" [structural anywhere]
      rule cuda-get-error-string(19) => "texture is not bound to a pointer" [structural anywhere]
      rule cuda-get-error-string(20) => "invalid channel descriptor" [structural anywhere]
      rule cuda-get-error-string(21) => "invalid copy direction for memcpy" [structural anywhere]
      rule cuda-get-error-string(22) => "invalid address of constant" [structural anywhere]
      rule cuda-get-error-string(23) => "fetch from texture failed" [structural anywhere]
      rule cuda-get-error-string(24) => "cannot fetch from a texture that is not bound" [structural anywhere]
      rule cuda-get-error-string(25) => "incorrect use of __syncthreads()" [structural anywhere]
      rule cuda-get-error-string(26) => "linear filtering not supported for non-float type" [structural anywhere]
      rule cuda-get-error-string(27) => "read as normalized float not supported for 32-bit non float type" [structural anywhere]
      rule cuda-get-error-string(28) => "device emulation mode and device execution mode cannot be mixed" [structural anywhere]
      rule cuda-get-error-string(29) => "unload of CUDA runtime failed" [structural anywhere]
      rule cuda-get-error-string(30) => "unknown error" [structural anywhere]
      rule cuda-get-error-string(31) => "feature not yet implemented" [structural anywhere]
      rule cuda-get-error-string(32) => "memory size or pointer value too large to fit in 32 bit" [structural anywhere]
      rule cuda-get-error-string(33) => "invalid resource handle" [structural anywhere]
      rule cuda-get-error-string(34) => "device not ready" [structural anywhere]
      rule cuda-get-error-string(35) => "CUDA driver version is insufficient for CUDA runtime version" [structural anywhere]
      rule cuda-get-error-string(36) => "cannot set while device is active in this process" [structural anywhere]
      rule cuda-get-error-string(37) => "invalid surface reference" [structural anywhere]
      rule cuda-get-error-string(38) => "no CUDA-capable device is detected" [structural anywhere]
      rule cuda-get-error-string(39) => "uncorrectable ECC error encountered" [structural anywhere]
      rule cuda-get-error-string(40) => "shared object symbol not found" [structural anywhere]
      rule cuda-get-error-string(41) => "shared object initialization failed" [structural anywhere]
      rule cuda-get-error-string(42) => "limit is not supported on this architecture" [structural anywhere]
      rule cuda-get-error-string(43) => "duplicate global variable looked up by string name" [structural anywhere]
      rule cuda-get-error-string(44) => "duplicate texture looked up by string name" [structural anywhere]
      rule cuda-get-error-string(45) => "duplicate surface looked up by string name" [structural anywhere]
      rule cuda-get-error-string(46) => "all CUDA-capable devices are busy or unavailable" [structural anywhere]
      rule cuda-get-error-string(47) => "device kernel image is invalid" [structural anywhere]
      rule cuda-get-error-string(48) => "no kernel image is available for execution on the device" [structural anywhere]
      rule cuda-get-error-string(49) => "incompatible driver contxt" // TODO: contxt for context keeps K from spazzing. 
            [structural anywhere]
      rule cuda-get-error-string(50) => "peer access is already enabled" [structural anywhere]
      rule cuda-get-error-string(51) => "peer access has not been enabled" [structural anywhere]
      rule cuda-get-error-string(52) => "unspecified driver error" [structural anywhere]
      rule cuda-get-error-string(53) => "unspecified driver error" [structural anywhere]
      rule cuda-get-error-string(54) => "exclusive-thread device already in use by a different thread" [structural anywhere]
      rule cuda-get-error-string(55) => "Profiling APIs disabled: application running with profiling tool." [structural anywhere]
      rule cuda-get-error-string(56) => "Profiler not initialized: call cudaProfilerInitialize()" [structural anywhere]
      rule cuda-get-error-string(57) => "Profiler already started." [structural anywhere]
      rule cuda-get-error-string(58) => "Profiler already stopped." [structural anywhere]
      rule cuda-get-error-string(59) => "device-side assert triggered" [structural anywhere]
      rule cuda-get-error-string(60) => "peer mapping resources exhausted" [structural anywhere]
      rule cuda-get-error-string(61) => "part or all of the requested memory range is already mapped" [structural anywhere]
      rule cuda-get-error-string(62) => "pointer does not correspond to a registered memory region" [structural anywhere]
      rule cuda-get-error-string(63) => "OS call failed or operation not supported on this OS" [structural anywhere]
      rule cuda-get-error-string(Error:Nat) => "unspecified driver error" 
            when Error >Int 63 andBool Error <Int 127 [structural anywhere]
      rule cuda-get-error-string(127) => "startup failure in cuda runtime" [structural anywhere]
      rule cuda-get-error-string(10000) => "api failure base" [structural anywhere]

      rule [cuda-return-success]:
            <k> cuda-return-success
                  => tv(cuda-success, t(.Set, int))
            ...</k>
            <cudaLastError> _ => cuda-success </cudaLastError>

      rule [cuda-return]:
            <k> cuda-return(Error:Nat)
                  => tv(Error, t(.Set, int))
            ...</k>
            <cudaLastError> _ => Error </cudaLastError>

      rule [cudaGetLastError]:
            <k> prepareBuiltin((Identifier("cudaGetLastError")), _)
                  => tv(LastError:Nat, t(.Set, int))
            ...</k>
            <cudaLastError> LastError => cuda-success </cudaLastError>

      rule [cudaPeekAtLastError]:
            <k> prepareBuiltin((Identifier("cudaPeekAtLastError")), _)
                  => tv(LastError:Nat, t(.Set, int))
            ...</k>
            <cudaLastError> LastError </cudaLastError>

      rule [cudaGetErrorString]:
            <k> prepareBuiltin((Identifier("cudaGetErrorString")), tv(Error:Nat, _))
                  => Constant(StringLiteral(cuda-get-error-string(Error)))
            ...</k>
end module
