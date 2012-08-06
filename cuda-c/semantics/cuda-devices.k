load cuda-helpers
load cuda-headers
load dynamic-c-semantics

module CUDA-DEVICES
      including CUDA-DEVICES-HEADER
      including CUDA-HELPERS
      including CUDA-RACECHECK-HEADER
      including DYNAMIC-C-SEMANTICS

      rule [cudaGetDeviceCount]:
            <k> prepareBuiltin((Identifier("cudaGetDeviceCount")), PtrCount:KResult)
                  => Computation(*(PtrCount) := tv(1, t(.Set, int)))
                  ~> cuda-success
            ...</k>

      rule [cudaDeviceCanAccessPeer]:
            <k> prepareBuiltin((Identifier("cudaDeviceCanAccessPeer")), (PtrAccess:KResult,, tv(Device:Nat, _),, tv(PeerDevice:Nat, _)))
                  => Computation(*(PtrAccess) := tv(0, t(.Set, int)))
                  ~> cuda-success
            ...</k>
            when Device:Nat ==Int 1
            andBool PeerDevice:Nat ==Int 1
      rule [cudaDeviceCanAccessPeer-invalid]:
            <k> prepareBuiltin((Identifier("cudaDeviceCanAccessPeer")), (_,, tv(Device:Nat, _),, tv(PeerDevice:Nat, _)))
                  => cuda-error-invalid-device
            ...</k>
            when Device:Nat =/=Int 1
            orBool PeerDevice:Nat =/=Int 1

      rule [cudaDeviceDisablePeerAccess]:
            <k> prepareBuiltin((Identifier("cudaDeviceDisablePeerAccess")), tv(PeerDevice:Nat, _))
                  => cuda-error-peer-access-not-enabled
            ...</k>
            when PeerDevice:Nat ==Int 1
      rule [cudaDeviceDisablePeerAccess-invalid]:
            <k> prepareBuiltin((Identifier("cudaDeviceDisablePeerAccess")), tv(PeerDevice:Nat, _))
                  => cuda-error-invalid-device
            ...</k>
            when PeerDevice:Nat =/=Int 1

      rule [cudaDeviceEnablePeerAccess]:
            <k> prepareBuiltin((Identifier("cudaDeviceEnablePeerAccess")), _)
                  => cuda-error-invalid-device
            ...</k>

      rule [cudaDeviceReset]:
            <k> prepareBuiltin((Identifier("cudaDeviceReset")), _)
                  => cuda-success
            ...</k>

      /*@ Device synchronization. */

      rule [cudaDeviceSynchronize]:
            <k> prepareBuiltin((Identifier("cudaDeviceSynchronize")), _) => cuda-device-synchronize ...</k>
      rule [cuda-device-synchronize]:
           <k> cuda-device-synchronize => cuda-rc-sync-global ~> cuda-success ...</k>
           <activeStreams> .Set </activeStreams> // i.e., all streams are empty.

end module
