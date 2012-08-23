load dynamic-c-semantics
load cuda-headers

module CUDA-EXECUTION is
      including DYNAMIC-C-SEMANTICS
      including CUDA-ERRORS-HEADER
      including CUDA-PROPERTIES-HEADER

      rule [cudaSetDoubleForDevice]:
            <k> prepareBuiltin((Identifier("cudaSetDoubleForDevice")), _)
                  => cuda-return-success
            ...</k>

      rule [cudaSetDoubleForHost]:
            <k> prepareBuiltin((Identifier("cudaSetDoubleForHost")), _)
                  => cuda-return-success
            ...</k>

      rule [cudaDriverGetVersion]:
            <k> prepareBuiltin((Identifier("cudaDriverGetVersion")), (PtrVersion:KResult))
                  => Computation(*(PtrVersion) := tv(cuda-cfg-driverVersion, t(.Set, int)))
                  ~> cuda-return-success
            ...</k>
      rule [cudaDriverGetVersion-null]:
            <k> prepareBuiltin((Identifier("cudaDriverGetVersion")), (tv(NullPointer, _)))
                  => cuda-error-invalid-value
            ...</k>

      rule [cudaRuntimeGetVersion]:
            <k> prepareBuiltin((Identifier("cudaRuntimeGetVersion")), (PtrVersion:KResult))
                  => Computation(*(PtrVersion) := tv(cuda-cfg-runtimeVersion, t(.Set, int)))
                  ~> cuda-return-success
            ...</k>
      rule [cudaRuntimeGetVersion-null]:
            <k> prepareBuiltin((Identifier("cudaRuntimeGetVersion")), (tv(NullPointer, _)))
                  => cuda-error-invalid-value
            ...</k>
end module
