load common-c-semantics

module CUDA-HELPERS is
      including COMMON-SEMANTIC-SYNTAX

      ops   cuda-success 
            cuda-error-invalid-resource-handle 
            cuda-error-not-ready 
            : -> Nat
      macro cuda-success = tv(0, t(.Set, int))
      macro cuda-error-invalid-resource-handle = tv(33, t(.Set, int))
      macro cuda-error-not-ready = tv(34, t(.Set, int))

      ops cuda-dim3-type : -> Type
      macro cuda-dim3-type = t(.Set, structType(Identifier("dim3")))
end module
