load dynamic-c-semantics

module CUDA-TYPING
      including COMMON-C-TYPING

      rule <k> canonicalizeType-aux(B:Bag (BagItem(T:K) => .Bag), _, _, _, (_ .Bag => BagItem(T:K))) ...</k>
            when T:K ==K CudaGlobal
            orBool T:K ==K CudaDevice
            orBool T:K ==K CudaHost
            [structural]
      rule <k> canonicalizeType-aux(.Bag, (T:KResult => t(.Set, qualifiedType(T:KResult, Q:K))), .Bag, .Bag, (_ BagItem(Q:K) => .Bag)) ...</k>
            when Q:K ==K CudaGlobal
            orBool Q:K ==K CudaDevice
            orBool Q:K ==K CudaHost
            [structural]

   //   rule [ignore-cuda-global]: t(S:Set, qualifiedType(t(S':Set, T:K), CudaGlobal)) 
   //         => t(S:Set S':Set, T:K) 
   //         [structural anywhere]
   //   rule [ignore-cuda-device]: t(S:Set, qualifiedType(t(S':Set, T:K), CudaDevice)) 
   //         => t(S:Set S':Set, T:K) 
   //         [structural anywhere]
      rule [ignore-cuda-host]: t(S:Set, qualifiedType(t(S':Set, T:K), CudaHost)) 
            => t(S:Set S':Set, T:K) 
            [structural anywhere]


      declare isCudaDeviceType : Type -> Bool
      declare isCudaGlobalType : Type -> Bool

      define [isCudaDeviceType-true]: isCudaDeviceType(t(_, qualifiedType(_, CudaDevice))) => true 
      define [isCudaDeviceType-prototype]: isCudaDeviceType(t(_, prototype(T:KResult))) => isCudaDeviceType(T:KResult) 
      define [isCudaDeviceType-qualified]: isCudaDeviceType(t(_, qualifiedType(T:KResult, Q:K))) => isCudaDeviceType(T:KResult) 
            when getKLabel(Q:K) =/=KLabel 'CudaDevice
      define [isCudaDeviceType-false]: isCudaDeviceType(t(_, T:K)) => false
            when getKLabel(T:K) =/=KLabel 'functionType
            andBool getKLabel(T:K) =/=KLabel 'qualifiedType
            andBool getKLabel(T:K) =/=KLabel 'prototype 

      define [isCudaGlobalType-true]: isCudaGlobalType(t(_, qualifiedType(_, CudaGlobal))) => true 
      define [isCudaGlobalType-prototype]: isCudaGlobalType(t(_, prototype(T:KResult))) => isCudaGlobalType(T:KResult) 
      define [isCudaGlobalType-qualified]: isCudaGlobalType(t(_, qualifiedType(T:KResult, Q:K))) => isCudaGlobalType(T:KResult) 
            when getKLabel(Q:K) =/=KLabel 'CudaGlobal
      define [isCudaGlobalType-false]: isCudaGlobalType(t(_, T:K)) => false
            when getKLabel(T:K) =/=KLabel 'functionType
            andBool getKLabel(T:K) =/=KLabel 'qualifiedType
            andBool getKLabel(T:K) =/=KLabel 'prototype 

    //  define [isCudaHostType-true]: isCudaHostType(t(_, qualifiedType(_, CudaHost))) => true 
    //  define [isCudaHostType-prototype]: isCudaHostType(t(_, prototype(T:KResult))) => isCudaHostType(T:KResult) 
    //  define [isCudaHostType-qualified]: isCudaHostType(t(_, qualifiedType(T:KResult, Q:K))) => isCudaHostType(T:KResult) 
    //        when getKLabel(Q:K) =/=KLabel 'CudaHost
    //  define [isCudaHostType-false]: isCudaHostType(t(_, T:K)) => false
    //        when getKLabel(T:K) =/=KLabel 'functionType
    //        andBool getKLabel(T:K) =/=KLabel 'qualifiedType
    //        andBool getKLabel(T:K) =/=KLabel 'prototype 

      // Actually seems like this should just be void.
      rule <type> CudaSpawn2(T:KResult, _, _) => innerType(T) ...</type>
      rule <type> CudaSpawn3(T:KResult, _, _, _) => innerType(T) ...</type>
      rule <type> CudaSpawn4(T:KResult, _, _, _, _) => innerType(T) ...</type>
end module

module CUDA-SEMANTICS is
      including DYNAMIC-C-SEMANTICS
      including CUDA-TYPING

      // call, nblocks, threads per block, nshared, stream
      syntax K ::= "cuda-launch-kernel" "(" K "," Nat "," Nat "," Nat "," Nat ")"
                  // grid id, blocks left to spawn, threads left to spawn.
                  | "cuda-launch-threads" "(" Nat "," Nat "," Nat ")"
                  | "cuda-launch-thread" "(" K "," Nat "," Nat "," Nat ")"
                  // grid id
                  | "cuda-next-grid" "(" Nat ")"
                  // call, nblocks, nthreads, nshared
                  | "cuda-grid" "(" K "," Nat "," Nat "," Nat ")"
                  | "cuda-join" "(" Nat ")"
                  | "cuda-sync-grid"
                  //          token, grid id, block id, this id, total ids
                  | "cuda-sync" "(" Nat "," Nat "," Nat "," Nat "," Nat ")"

      // TODO: Somehow check that function has the "CudaGlobal" qualified
      // type. This is difficult because it somehow gets stripped (via
      // unqualifyType?) from the type of Fun.
      rule [cuda-spawn2]:
           <k> CudaSpawn2(Fun:K, tv(NBlocks:Nat, _), tv(NThreads:Nat, _))
                 => cuda-launch-kernel(Fun, NBlocks, NThreads, 0, 0)
           ...</k>
           <gid> 0 </gid>
           [structural]

      rule [cuda-spawn3]:
           <k> CudaSpawn3(Fun:K, tv(NBlocks:Nat, _), tv(NThreads:Nat, _), tv(NShared:Nat, _))
                 => cuda-launch-kernel(Fun, NBlocks, NThreads, NShared, 0)
           ...</k>
           <gid> 0 </gid>
           [structural]

      rule [cuda-spawn4]:
           <k> CudaSpawn4(Fun:K, tv(NBlocks:Nat, _), tv(NThreads:Nat, _), tv(NShared:Nat, _), tv(SId:Nat, _))
                 => cuda-launch-kernel(Fun, NBlocks, NThreads, NShared, SId)
           ...</k>
           <gid> 0 </gid>
           [structural]

      // This is to get ThreadIdx and the like into the environment.
      // TODO: there must be a better way.
// rule [function-application-pre]:
// 	<k> Call(tv(Loc:Nat, t(_, pointerType(T:KResult))), List(L:List{KResult}))
// 		=> application(readFunction(Loc:Nat), L:List{KResult})
// 	...</k>
// 	when isFunctionType(T:KResult)
// 	[structural]
// rule [function-application]:
// 	<k> application(functionObject(X:Id, t(_, functionType(R:KResult, P:List{KResult})), B:K), L:List{KResult}) ~> K:K 
// 		=> sequencePoint
// 		~> populateFromGlobal
// 		~> bind(L:List{KResult}, P:List{KResult})
// 		~> B:K
// 	</k>
// 	<callStack> .List => ListItem(
// 		<stackFrame> C:Bag
// 			<continuation> K:K </continuation>
// 			<currentTranslationUnit> OldTu:K </currentTranslationUnit>
// 			<stackCurrentFunction> CurrFun:K </stackCurrentFunction>
// 			<stackCurrentProgramLoc> CurrLoc:K </stackCurrentProgramLoc>
// 		</stackFrame>
// 	) ...</callStack>
// 	<br/>
// 	(<control>
// 		<currentTranslationUnit> OldTu:K </currentTranslationUnit>
// 		<currentFunction> CurrFun:K </currentFunction>
// 		<currentProgramLoc> CurrLoc:K </currentProgramLoc>
// 		C:Bag
// 	</control>
// 	=>
// 	<control>
// 		<currentFunction> X:Id </currentFunction>
// 		<currentTranslationUnit> Tu:K </currentTranslationUnit>
// 		<currentProgramLoc> CurrLoc:K </currentProgramLoc>
// 	...</control>)
// 	<functionTranslationUnits>... OldTu:K |-> Map((_ X:Id |-> Tu:K)) ...</functionTranslationUnits>
// 	[structural large]
// 
      rule < k => finalComputation > cuda-launch-kernel(_, 0, _, _, _) ...</ k => finalComputation > 
           (.Bag => <errorCell> Error("90000", "NBlocks zero.") </errorCell>)
      rule < k => finalComputation > cuda-launch-kernel(_, _, 0, _, _) ...</ k => finalComputation > 
           (.Bag => <errorCell> Error("90001", "NThreads zero.") </errorCell>)

      ops cuda-launch-success cuda-sync-success : -> Nat
      macro cuda-launch-success = tv(1, t(.Set, int))
      macro cuda-sync-success = tv(1, t(.Set, int))

      // TODO: return value.
      rule <k> cuda-launch-kernel(FCall:K, NBlocks:Nat, NThreads:Nat, NShared:Nat, SId:Nat) => cuda-launch-success ...</k>
           <nextGid> GId:Nat => GId +Int 1 </nextGid> 
           <grids>... (. => GId |-> cuda-grid(FCall, NBlocks, NThreads, NShared)) ...</grids>
           <stream>... <sid> SId </sid> 
                 <streamContents> ... (. => cuda-next-grid(GId)) </streamContents> 
                 ...</stream>
      rule <k> cuda-launch-kernel(FCall:K, NBlocks:Nat, NThreads:Nat, NShared:Nat, SId:Nat) => cuda-launch-success ...</k>
           <nextGid> GId:Nat => GId +Int 1 </nextGid> 
           <grids>... (. => GId |-> cuda-grid(FCall, NBlocks, NThreads, NShared)) ...</grids>
           <streams> Streams:Set (. => SetItem(SId)) </streams>
           (. => <stream>... <sid> SId </sid> 
                 <streamContents> cuda-next-grid(GId) </streamContents> 
                 ...</stream>)
           when notBool(SId in Streams)

      rule <streamContents> cuda-next-grid(GId:Nat) =>
            cuda-launch-threads(GId, NBlocks -Int 1, NThreads -Int 1) ...</streamContents>
            <grids>... GId |-> cuda-grid(_, NBlocks:Nat, NThreads:Nat, _) ...</grids>
 
       rule [cuda-launch-thread]:
            <streamContents> cuda-launch-thread(TBody:K, GId:Nat, BId:Nat, TId:Nat) => . ...</streamContents>
            // TODO: must we clone the main thread?
            <nextThreadId> ThreadId:Nat => ThreadId +Int 1</nextThreadId>
            <threadStatus> Status:Map => Status[threadRunning / ThreadId] </threadStatus>
            <thread>
                  <gid> 0 </gid>
                  <bid> 0 </bid>
                  <tid> 0 </tid>
                  <nextLoc> _:Nat </nextLoc>
                  <threadId> _:Nat </threadId>
                  C:Bag
                  <k> _:K </k>
                  <threadLocal>
                        <callStack> _:List </callStack>
                        C'':Bag
                        <control>
                              C':Bag
                        </control>
                  </threadLocal>
            </thread>
            (.Bag =>
            <thread>
                  <gid> GId </gid>
                  <bid> BId </bid>
                  <tid> TId </tid> 
                  <nextLoc> loc(threadId(ThreadId) +Int 0, 0, 0) </nextLoc>
                  <threadId> ThreadId </threadId>
                  C:Bag
                  <k> TBody </k>
                  <threadLocal>
                        <callStack> .List </callStack>
                        <control>
                              C':Bag
                        </control>
                  </threadLocal>
            </thread>)
            [computational]
 
       /*@ Launch the first thread of a grid. This thread is responsible for
        clearing the stream when it's done executing.*/

       rule [cuda-launch-grid-head]:
            <streamContents> 
                  (. => cuda-launch-thread((Computation(FCall) ~> cuda-sync-grid ~> cuda-join(GId)), GId, 0, 0))
                  ~> (cuda-launch-threads(GId:Nat, 0, 0) => cuda-join(GId))
            ...</streamContents>
            <grids> ... GId |-> cuda-grid(FCall:K, NBlocks:Nat, NThreads:Nat, NShared:Nat) ... </grids>
            [structural]

      /*@ Launch the first thread of a block. */

      rule [cuda-launch-block-head]:
           <streamContents> 
                  (. => cuda-launch-thread((Computation(FCall) ~> cuda-sync-grid), GId, BId, 0)) 
                  ~> cuda-launch-threads(GId:Nat, (BId:Nat => BId -Int 1), (0 => NThreads -Int 1)) 
            ...</streamContents>
           <grids> ... GId |-> cuda-grid(FCall:K, NBlocks:Nat, NThreads:Nat, NShared:Nat) ... </grids>
           when BId >Int 0
           [structural]

      /*@ Launch the other threads. */

      rule [cuda-launch-other-threads]:
           <streamContents> 
                  (. => cuda-launch-thread((Computation(FCall) ~> cuda-sync-grid), GId, BId, TId)) 
                  ~> cuda-launch-threads(GId:Nat, BId:Nat, (TId:Nat => TId -Int 1)) 
            ...</streamContents>
           <grids> ... GId |-> cuda-grid(FCall:K, NBlocks:Nat, NThreads:Nat, NShared:Nat) ... </grids>
           when TId >Int 0
           [structural]

      rule <k> Identifier("threadIdx") => tv(TId:Nat, t(.Set, int)) ...</k>
           <tid> TId:Nat </tid> 
      rule <k> Identifier("blockIdx") => tv(BId:Nat, t(.Set, int)) ...</k>
           <bid> BId:Nat </bid>
      rule <k> Identifier("gridDim") => tv(NBlocks:Nat, t(.Set, int)) ...</k>
           <gid> GId:Nat </gid>
           <grids> ... GId |-> cuda-grid(_, NBlocks:Nat, _, _) ... </grids>
      rule <k> Identifier("blockDim") => tv(NThreads:Nat, t(.Set, int)) ...</k>
           <gid> GId:Nat </gid>
           <grids> ... GId |-> cuda-grid(_, _, NThreads:Nat, _) ... </grids>

      /*@ Clear the stream so the next grid can execute. */

      rule <streamContents> cuda-join(GId:Nat) => . ...</streamContents>
            <k> cuda-join(GId) => . ...</k>
  
      /*@ \subsection{Block-level synchronization} */

      /*@ Start the token at thread 0. */

      // TODO: prepareBuiltin?
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <thread>... <tid> 0 </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid> 
            <k> prepareBuiltin((Identifier("__syncthreads")), _) => cuda-sync(1, GId, BId, 0, NThreads) ...</k> ...</thread>
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <thread>... <tid> TId:Nat </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid>
            <k> prepareBuiltin((Identifier("__syncthreads")), _) => cuda-sync(0, GId, BId, TId, NThreads) ...</k> ...</thread>
            when TId >Int 0

      /*@ Pass the token up. */

      rule [cuda-syncthreads-passup]:
            <k> cuda-sync(1 => 0, GId:Nat, BId:Nat, TId:Nat, NThreads:Nat) ...</k>
            <k> cuda-sync(0 => 1, GId, BId, SuccTId:Nat, NThreads) ...</k>
            when SuccTId ==Int TId +Int 1

      /*@ Pivot. */

      rule [cuda-syncthreads-pivot]:
            <k> cuda-sync(1 => 2, _, _, TId:Nat, NThreads:Nat) ...</k> 
            when NThreads ==Int TId +Int 1

      /*@ Pass it back down. */

      // TODO: return value.
      rule [cuda-syncthreads-passdown]:
            <k> cuda-sync(2, GId:Nat, BId:Nat, SuccTId:Nat, NThreads:Nat) => cuda-sync-success...</k>
            <k> cuda-sync(0 => 2, GId, BId, TId:Nat, NThreads) ...</k>
            when SuccTId ==Int TId +Int 1
      rule <k> cuda-sync(2, _, _, 0, _) => cuda-sync-success ...</k> 

      /*@ \subsection{Grid-level synchronization} 
      This isn't an operation supported by CUDA, but we need it in this model to make
      the streams synchronous. Same deal as syncing threads within a block. This
      shouldn't ever clash with a block-level sync because if there are multiple
      blocks in the grid (the only time a clash would be possible), then NThreads
      will be different for the grid-level and block-level sync.*/

      rule <grids>... GId |-> cuda-grid(_, NBlocks:Nat, NThreads:Nat, _) ...</grids>
            <thread>... <gid> GId:Nat </gid> <bid> 0 </bid> <tid> 0 </tid>
            <k> cuda-sync-grid => Computation(cuda-sync(1, GId, 0, 0, NThreads *Int NBlocks)) ...</k> ...</thread>
      rule <grids>... GId |-> cuda-grid(_, NBlocks:Nat, NThreads:Nat, _) ...</grids>
            <thread>... <gid> GId:Nat </gid> <bid> BId:Nat </bid> <tid> TId:Nat </tid>
            <k> cuda-sync-grid => Computation(cuda-sync(0, GId, 0, (NThreads *Int BId) +Int TId, NThreads *Int NBlocks)) ...</k> ...</thread>
            when BId >Int 0 orBool TId >Int 0


      // Cleanup completed threads. 
      // TODO: probably other stuff needs to be done here (free memory,
      // perhaps?)
      rule (<thread>... <k>.K</k> ...</thread> => .)

      /*@ \subsection{Stream Management} */

      /*@ Device synchronization. */
      rule <k> prepareBuiltin((Identifier("cudaDeviceSynchronize")), _) => cuda-sync-success ...</k>
            <streams> .Set </streams> // i.e., all streams are empty.

      /*@ Stream synchronization. */
      // TODO: return value (also argument).
      rule <k> prepareBuiltin((Identifier("cudaStreamSynchronize")), tv(SId:Nat, t(_, int))) 
                  => tv(1, t(.Set, int)) ...</k>
            <streams> Streams:Set </streams>
            when notBool(SId in Streams)

      /*@ Remove defunct streams. */
      rule (<stream>... <sid>SId:Nat</sid> 
            <streamContents> .K </streamContents> ...</stream> => .)
            <streams> ... (SetItem(SId)=>.) ...</streams>

end module


