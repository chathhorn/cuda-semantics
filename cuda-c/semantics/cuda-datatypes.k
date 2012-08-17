load common-c-semantics
load cuda-headers

module CUDA-DATATYPES is
      including COMMON-SEMANTIC-SYNTAX
      including CUDA-DATATYPES-HEADER

      macro cuda-dim3-type = t(.Set, qualifiedType(t(.Set, structType(Identifier("dim3"))), CudaDevice))
end module
