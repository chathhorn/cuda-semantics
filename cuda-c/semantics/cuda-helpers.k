load common-c-semantics

module CUDA-HELPERS is
      including COMMON-SEMANTIC-SYNTAX

      ops   cuda-success 
            cuda-error-invalid-device
            cuda-error-invalid-resource-handle 
            cuda-error-not-ready 
            cuda-error-peer-access-not-enabled
            : -> Nat
      macro cuda-success = tv(0, t(.Set, int))
      macro cuda-error-invalid-device = tv(10, t(.Set, int))
      macro cuda-error-invalid-resource-handle = tv(33, t(.Set, int))
      macro cuda-error-not-ready = tv(34, t(.Set, int))
      macro cuda-error-peer-access-not-enabled = tv(51, t(.Set, int))

      ops cuda-dim3-type : -> Type
      macro cuda-dim3-type = t(.Set, qualifiedType(t(.Set, structType(Identifier("dim3"))), CudaDevice))
end module
