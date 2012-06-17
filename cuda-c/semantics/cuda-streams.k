load cuda-helpers
load cuda-headers
load dynamic-c-semantics

module CUDA-STREAMS
      including CUDA-STREAMS-HEADER
      including CUDA-HELPERS
      including CUDA-DEVICES-HEADER
      including DYNAMIC-C-SEMANTICS

      syntax K ::= "cuda-put-in-stream-aux" "(" K "," Nat ")" 

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

      // TODO: I still don't understand the expected behavior of the 0 stream.
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

      /*@ Put something in a stream. We synchronize the device before inserting
      anything into stream zero.*/

      rule [cuda-put-in-stream-zero]:
            <k> cuda-put-in-stream(Contents:K, 0)
                  => Computation(cuda-device-synchronize)
                  ~> cuda-put-in-stream-aux(Contents:K, 0)
            ...</k>
      rule [cuda-put-in-stream]:
            <k> cuda-put-in-stream(Contents:K, SId:Nat)
                  => cuda-put-in-stream-aux(Contents:K, SId:Nat)
            ...</k>
            when SId =/=Int 0

      rule [cuda-put-in-existing-stream]:
            <k> cuda-put-in-stream-aux(Contents:K, SId:Nat) => . ...</k>
            <initializedStreams>... SetItem(SId:Nat) ...</initializedStreams>
            <stream>... 
                  <sid> SId </sid> 
                  <streamContents> ... (. => Contents) </streamContents> 
            ...</stream>
      [structural]
      rule [cuda-put-in-new-stream]:
           <k> cuda-put-in-stream-aux(Contents:K, SId:Nat) => . ...</k>
           <initializedStreams> InitializedStreams:Set </initializedStreams>
           <activeStreams> ActiveStreams:Set (. => SetItem(SId)) </activeStreams>
           (. => <stream>... <sid> SId </sid> 
                 <streamContents> Contents </streamContents> 
            ...</stream>)
           when SId in InitializedStreams:Set
           andBool notBool(SId in ActiveStreams:Set)
           [structural]
      rule [cuda-put-in-uninitialized-stream]:
           < k => finalComputation > cuda-put-in-stream-aux(Contents:K, SId:Nat) 
           ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90401", "CUDA: Attempting to use an uncreated stream.") </errorCell>)
           <initializedStreams> Streams:Set </initializedStreams>
           when notBool(SId in Streams)

      /*@ Stream synchronization. */

      syntax K ::= "cuda-stream-synchronize" "(" Nat ")"
      rule [cudaStreamSynchronize]:
            <k> prepareBuiltin((Identifier("cudaStreamSynchronize")), tv(SId:Nat, _)) 
                  => cuda-stream-synchronize(SId:Nat) 
            ...</k>
      rule [cuda-stream-synchronize]:
            <k> cuda-stream-synchronize(SId:Nat) 
                  => cuda-success 
            ...</k>
            <initializedStreams>... SetItem(SId) ...</initializedStreams>
            <activeStreams> Streams:Set </activeStreams>
            when notBool(SId in Streams)
      rule [cuda-stream-synchronize-error]:
            <k> cuda-stream-synchronize(SId:Nat) 
                  => cuda-error-invalid-resource-handle 
            ...</k>
            <initializedStreams> Initialized:Set </initializedStreams>
            when notBool(SId in Initialized)

      /*@ Wait for an event to be recorded in a stream. */

      syntax K ::= "cuda-stream-wait-event" "(" Nat ")"

      // TODO: Flags ("must be 0").
      rule [cudaStreamWaitEvent-existing]:
            <k> prepareBuiltin((Identifier("cudaStreamWaitEvent")), (tv(SId:Nat, _),,tv(EId:Nat, _),,tv(Flags:Nat, _)))
                  => cuda-put-in-stream(cuda-stream-wait-event(EId:Nat), SId:Nat)
                  ~> cuda-success
            ...</k>
            <initializedEvents>... SetItem(EId:Nat) ...</initializedEvents>
            <initializedStreams>... SetItem(SId:Nat) ...</initializedStreams>
      rule [cudaStreamWaitEvent-non-existing]:
            <k> prepareBuiltin((Identifier("cudaStreamWaitEvent")), (tv(SId:Nat, _),,tv(EId:Nat, _),,tv(Flags:Nat, _)))
                  => cuda-error-invalid-resource-handle
            ...</k>
            <initializedEvents> InitializedEvents:Set </initializedEvents>
            <initializedStreams> InitializedStreams:Set </initializedStreams>
            when notBool (EId:Nat in InitializedEvents:Set)
            orBool notBool (SId:Nat in InitializedStreams:Set)

      rule [cuda-stream-wait-event]:
            <streamContents> cuda-stream-wait-event(EId:Nat) 
                  => .
            ...</streamContents>
            <pendingEvents>... EId:Nat |-> PId:Nat ...</pendingEvents>
            <recordedEvents>... EId:Nat |-> PId:Nat ...</recordedEvents>
      rule [cuda-stream-wait-event-not-pending]:
            <streamContents> cuda-stream-wait-event(EId:Nat) 
                  => .
            ...</streamContents>
            <pendingEvents> Pending:Map </pendingEvents>
            when notBool (EId:Nat in keys Pending)

      /*@ Remove defunct streams. */

      rule (<stream>... <sid>SId:Nat</sid> 
            <streamContents> .K </streamContents> ...</stream> => .)
            <activeStreams> ... (SetItem(SId)=>.) ...</activeStreams>

      /*@ Events. */

      rule [cudaEventCreate]:
            <k> prepareBuiltin((Identifier("cudaEventCreate")), ReturnPtrLoc:KResult)
                  => Computation(*(ReturnPtrLoc) := tv(EId:Nat, t(.Set, int)))
                  ~> cuda-success
            ...</k>
            <nextEid> EId:Nat => EId +Int 1 </nextEid>
            <initializedEvents>... (. => SetItem(EId)) ...</initializedEvents>

      // TODO: Flags.
      rule [cudaEventCreateWithFlags]:
            <k> prepareBuiltin((Identifier("cudaEventCreateWithFlags")), (ReturnPtrLoc:KResult,,tv(Flags:Nat, _)))
                  => Computation(*(ReturnPtrLoc) := tv(EId:Nat, t(.Set, int)))
                  ~> cuda-success
            ...</k>
            <nextEid> EId:Nat => EId +Int 1 </nextEid>
            <initializedEvents>... (. => SetItem(EId)) ...</initializedEvents>

      /*@ Destroy event. */

      rule [cudaEventDestory-existing]:
            <k> prepareBuiltin((Identifier("cudaEventDestroy")), tv(EId:Nat, _))
                  => cuda-success
            ...</k>
            <initializedEvents>... SetItem(EId:Nat) => . ...</initializedEvents>
      rule [cudaEventDestory-non-existing]:
            <k> prepareBuiltin((Identifier("cudaEventDestroy")), tv(EId:Nat, _))
                => cuda-error-invalid-resource-handle
            ...</k>
            <initializedEvents> Initialized:Set </initializedEvents>
            when notBool(EId in Initialized)

      syntax K ::= "cuda-event-record" "(" Nat "," Nat ")"

      rule [cudaEventRecord-existing]:
            <k> prepareBuiltin((Identifier("cudaEventRecord")), (tv(EId:Nat, _),,tv(SId:Nat, _)))
                  => cuda-put-in-stream(cuda-event-record(EId:Nat, PId:Nat), SId:Nat)
                  ~> cuda-success
            ...</k>
            <nextPendingId> PId:Nat => PId:Nat +Int 1 </nextPendingId>
            <initializedEvents>... SetItem(EId:Nat) ...</initializedEvents>
            <pendingEvents> Pending:Map => Pending:Map[PId:Nat/EId:Nat] </pendingEvents>
            <recordedEvents> Recorded:Map => Recorded:Map[0/EId:Nat] </recordedEvents>
      rule [cudaEventRecord-non-existing]:
            <k> prepareBuiltin((Identifier("cudaEventRecord")), (tv(EId:Nat, _),,tv(SId:Nat, _)))
                => cuda-error-invalid-resource-handle
            ...</k>
            <initializedEvents> Initialized:Set </initializedEvents>
            when notBool(EId in Initialized)

      rule [cuda-event-record]:
            <streamContents> cuda-event-record(EId:Nat, PId:Nat) => . ...</streamContents>
            <pendingEvents>... EId:Nat |-> PId:Nat ...</pendingEvents>
            <recordedEvents>... EId:Nat |-> (_ => PId:Nat) ...</recordedEvents>
      rule [cuda-event-record-defunct]:
            <streamContents> cuda-event-record(EId:Nat, PId:Nat) => . ...</streamContents>
            <pendingEvents>... EId:Nat |-> CurrentPId:Nat ...</pendingEvents>
            when PId:Nat =/=Int CurrentPId:Nat

      rule [cudaEventQuery-recorded]:
            <k> prepareBuiltin((Identifier("cudaEventQuery")), tv(EId:Nat, _))
                  => cuda-success
            ...</k>
            <initializedEvents>... SetItem(EId:Nat) ...</initializedEvents>
            <pendingEvents>... EId:Nat |-> PId:Nat ...</pendingEvents>
            <recordedEvents>... EId:Nat |-> PId:Nat ...</recordedEvents>
      rule [cudaEventQuery-not-pending]:
            <k> prepareBuiltin((Identifier("cudaEventQuery")), tv(EId:Nat, _))
                  => cuda-success
            ...</k>
            <initializedEvents>... SetItem(EId:Nat) ...</initializedEvents>
            <pendingEvents> Pending:Map </pendingEvents>
            when notBool (EId:Nat in keys Pending:Map)
      rule [cudaEventQuery-existing]:
            <k> prepareBuiltin((Identifier("cudaEventQuery")), tv(EId:Nat, _))
                  => cuda-error-not-ready
            ...</k>
            <initializedEvents>... SetItem(EId:Nat) ...</initializedEvents>
            <pendingEvents>... EId:Nat |-> PId:Nat ...</pendingEvents>
            <recordedEvents>... EId:Nat |-> OldPId:Nat ...</recordedEvents>
            when PId:Nat =/=Int OldPId:Nat
      rule [cudaEventQuery-non-existing]:
            <k> prepareBuiltin((Identifier("cudaEventQuery")), tv(EId:Nat, _))
                => cuda-error-invalid-resource-handle
            ...</k>
            <initializedEvents> Initialized:Set </initializedEvents>
            when notBool(EId in Initialized)

      rule [cudaEventSynchronize-existing]:
            <k> prepareBuiltin((Identifier("cudaEventSynchronize")), tv(EId:Nat, _))
                  => cuda-success
            ...</k>
            <initializedEvents>... SetItem(EId:Nat) ...</initializedEvents>
            <pendingEvents> Pending:Map </pendingEvents>
            when notBool (EId:Nat in keys Pending:Map)
      rule [cudaEventSynchronize-recorded]:
            <k> prepareBuiltin((Identifier("cudaEventSynchronize")), tv(EId:Nat, _))
                  => cuda-success
            ...</k>
            <initializedEvents>... SetItem(EId:Nat) ...</initializedEvents>
            <pendingEvents>... EId:Nat |-> PId:Nat ...</pendingEvents>
            <recordedEvents>... EId:Nat |-> PId:Nat ...</recordedEvents>
      rule [cudaEventSynchronize-non-existing]:
            <k> prepareBuiltin((Identifier("cudaEventSynchronize")), tv(EId:Nat, _))
                => cuda-error-invalid-resource-handle
            ...</k>
            <initializedEvents> Initialized:Set </initializedEvents>
            when notBool(EId in Initialized)

      // TODO: need to check that StartEId & EndEId both have matching
      // Pending/Recorded PIds (if you catch my drift).
      rule [cudaEventElapsedTime-success]:
            <k> prepareBuiltin((Identifier("cudaEventElapsedTime")), (PtrMs:KResult,,tv(StartEId:Nat, _),,tv(EndEId:Nat, _)))
                  => Computation(*(PtrMs) := tv(0.0, t(.Set, float)))
                  ~> cuda-success
            ...</k>
            <initializedEvents> Initialized:Set </initializedEvents>
            <pendingEvents> Pending:Map </pendingEvents>
            <recordedEvents> Recorded:Map </recordedEvents>
            when StartEId:Nat in Initialized:Set
            andBool EndEId:Nat in Initialized:Set
            andBool StartEId:Nat in keys Recorded:Map
            andBool EndEId:Nat in keys Recorded:Map
      rule [cudaEventSynchronize-recorded]:
            <k> prepareBuiltin((Identifier("cudaEventElapsedTime")), (PtrMs:KResult,,tv(StartEId:Nat, _),,tv(EndEId:Nat, _)))
                  => cuda-error-not-ready
            ...</k>
            <initializedEvents> Initialized:Set </initializedEvents>
            <pendingEvents> Pending:Map </pendingEvents>
            <recordedEvents> Recorded:Map </recordedEvents>
            when StartEId:Nat in Initialized:Set
            andBool EndEId:Nat in Initialized:Set
            andBool (notBool StartEId:Nat in keys Recorded:Map
            orBool notBool EndEId:Nat in keys Recorded:Map)
      rule [cudaEventSynchronize-non-existing]:
            <k> prepareBuiltin((Identifier("cudaEventElapsedTime")), (PtrMs:KResult,,tv(StartEId:Nat, _),,tv(EndEId:Nat, _)))
                => cuda-error-invalid-resource-handle
            ...</k>
            <initializedEvents> Initialized:Set </initializedEvents>
            when notBool (StartEId in Initialized)
            orBool notBool (EndEId in Initialized)

end module
