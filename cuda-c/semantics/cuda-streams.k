load cuda-helpers
load dynamic-c-semantics

module CUDA-STREAMS
      including CUDA-HELPERS
      including DYNAMIC-C-SEMANTICS

      syntax K ::= "cuda-put-in-stream" "(" K "," Nat ")" 
                  | "cuda-join" "(" Nat ")"

      /*@ \subsection{Stream Management} */

      rule <streamContents> cuda-join(GId:Nat) => . ...</streamContents>
            <k> cuda-join(GId) => . ...</k>

      /*@ Create stream. */

      rule [cudaStreamCreate]:
            <k> prepareBuiltin((Identifier("cudaStreamCreate")), ReturnPtrLoc:KResult)
                  => Computation(*(ReturnPtrLoc) := tv(SId:Nat, t(.Set, int)))
                  ~> cuda-success
            ...</k>
            <nextSid> SId:Nat => SId +Int 1 </nextSid>
            <initializedStreams>... (. => SetItem(SId)) ...</initializedStreams>

      /*@ Stream 0 is forever legit. */

      rule <initializedStreams> Streams:Set (. => SetItem(0)) </initializedStreams>
            when notBool(0 in Streams:Set)

      rule [cudaStreamQuery-invalid]:
            <k> prepareBuiltin((Identifier("cudaStreamQuery")), tv(SId:Nat, _))
                  => cuda-error-invalid-resource-handle
            ...</k>
            <initializedStreams> Initialized:Set </initializedStreams>
            when notBool SId in Initialized
      rule [cudaStreamQuery-not-active]:
            <k> prepareBuiltin((Identifier("cudaStreamQuery")), tv(SId:Nat, _))
                  => cuda-success
            ...</k>
            <initializedStreams> Initialized:Set </initializedStreams>
            <activeStreams> Active:Set </activeStreams>
            when SId in Initialized
            andBool notBool SId in Active
      rule [cudaStreamQuery-active]:
            <k> prepareBuiltin((Identifier("cudaStreamQuery")), tv(SId:Nat, _))
                  => cuda-error-not-ready
            ...</k>
            <initializedStreams> Initialized:Set </initializedStreams>
            <activeStreams> Active:Set </activeStreams>
            when SId in Initialized
            andBool SId in Active

      /*@ Destroy stream. */

      rule [cudaStreamDestory-existing]:
            <k> prepareBuiltin((Identifier("cudaStreamDestroy")), tv(SId:Nat, _))
                  => cuda-success
            ...</k>
            <initializedStreams>... SetItem(SId:Nat) => . ...</initializedStreams>
      rule [cudaStreamDestory-non-existing]:
            <k> prepareBuiltin((Identifier("cudaStreamDestroy")), tv(SId:Nat, _))
                => cuda-error-invalid-resource-handle
            ...</k>
            <initializedStreams> Initialized:Set </initializedStreams>
            when notBool(SId in Initialized)

      rule [cuda-put-in-existing-stream]:
           <k> cuda-put-in-stream(Contents:K, SId:Nat) => . ...</k>
           <initializedStreams>... SetItem(SId:Nat) ...</initializedStreams>
           <stream>... 
                 <sid> SId </sid> 
                 <streamContents> ... (. => Contents) </streamContents> 
            ...</stream>
           [structural]
      rule [cuda-put-in-new-stream]:
           <k> cuda-put-in-stream(Contents:K, SId:Nat) => . ...</k>
           <initializedStreams> InitializedStreams:Set </initializedStreams>
           <activeStreams> ActiveStreams:Set (. => SetItem(SId)) </activeStreams>
           (. => <stream>... <sid> SId </sid> 
                 <streamContents> Contents </streamContents> 
            ...</stream>)
           when SId in InitializedStreams:Set
           andBool notBool(SId in ActiveStreams:Set)
           [structural]
      rule [cuda-put-in-uninitialized-stream]:
           < k => finalComputation > cuda-put-in-stream(Contents:K, SId:Nat) 
           ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90401", "CUDA: Attempting to use an uncreated stream.") </errorCell>)
           <initializedStreams> Streams:Set </initializedStreams>
           when notBool(SId in Streams)

      /*@ Stream synchronization. */

      syntax K ::= "cuda-stream-synchronize" "(" Nat ")"
      rule [cudaStreamSynchronize]:
            <k> prepareBuiltin((Identifier("cudaStreamSynchronize")), tv(SId:Nat, _)) 
                  => cuda-stream-synchronize(SId:Nat) ...</k>
      rule [cuda-stream-synchronize]:
            <k> cuda-stream-synchronize(SId:Nat) => cuda-success ...</k>
            <initializedStreams>... SetItem(SId) ...</initializedStreams>
            <activeStreams> Streams:Set </activeStreams>
            when notBool(SId in Streams)
      rule [cuda-stream-synchronize-error]:
            <k> cuda-stream-synchronize(SId:Nat) 
                  => cuda-error-invalid-resource-handle ...</k>
            <initializedStreams> Initialized:Set </initializedStreams>
            when notBool(SId in Initialized)

      /*@ Remove defunct streams. */

      rule (<stream>... <sid>SId:Nat</sid> 
            <streamContents> .K </streamContents> ...</stream> => .)
            <activeStreams> ... (SetItem(SId)=>.) ...</activeStreams>

      /*@ Device synchronization. */

      syntax K ::= "cuda-device-synchronize"
      rule <k> prepareBuiltin((Identifier("cudaDeviceSynchronize")), _) => cuda-device-synchronize ...</k>
      rule <k> cuda-device-synchronize => cuda-success ...</k>
           <activeStreams> .Set </activeStreams> // i.e., all streams are empty.

end module
