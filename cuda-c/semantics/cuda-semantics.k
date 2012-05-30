load dynamic-c-semantics
load cuda-typing

module CUDA-SEMANTICS is
      including DYNAMIC-C-SEMANTICS
      including CUDA-TYPING

      ops cuda-success : -> Nat
      macro cuda-success = tv(0, t(.Set, int))

      /*@ Declarations. */
       rule [cuda-declare-shared-dynamic]:
		<k> declareInternalVariable(X:Id, T:KResult, NoInit)
			=> declareWithLinkage(X:Id, T:KResult, SingleInit(tv(SharedLoc:Nat, unqualifyType(T:KResult))), noLinkage)
		...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaShared>... 
                  cuda-shared-loc(GId:Nat, BId:Nat, Identifier("$dynamic")) |-> SharedLoc:Nat 
            ...</cudaShared>
		when notBool isStaticType(T:KResult)
		andBool isExternType(T:KResult)
            andBool isCudaSharedType(T:KResult)
            // __shared__ and extern on anything but a pointer must be illegal.
            andBool (isPointerType(T:KResult) orBool isIncompleteType(T:KResult))
		[structural] 

       // TODO: Initializers (is that legal on shared variables?).
       rule [cuda-declare-shared-noalloc]:
		<k> declareInternalVariable(X:Id, T:KResult, NoInit)
			=> addToEnv(X:Id, SharedLoc:Nat)
			~> giveType(X:Id, T:KResult)
		...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaShared>... 
                  cuda-shared-loc(GId:Nat, BId:Nat, X:Id) |-> SharedLoc:Nat 
            ...</cudaShared>
		when notBool isStaticType(T:KResult)
		andBool notBool isExternType(T:KResult)
            andBool isCudaSharedType(T:KResult)
		[structural] 
       rule [cuda-declare-shared]:
		<k> declareInternalVariable(X:Id, T:KResult, NoInit)
			=> allocateType(loc(Base:Nat, 0, 0), T:KResult)
                  ~> cuda-set-mdevice(loc(Base:Nat, 0, 0)) 
                  ~> addToEnv(X:Id, loc(Base:Nat, 0, 0))
			~> giveType(X:Id, T:KResult)
		...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaShared>... 
                  . => (cuda-shared-loc(GId:Nat, BId:Nat, X:Id) |-> loc(Base:Nat, 0, 0))
            ...</cudaShared>
            <freshNat> Fresh:Nat => Fresh:Nat +Int 1 </freshNat>
		when notBool isStaticType(T:KResult)
		andBool notBool isExternType(T:KResult)
            andBool isCudaSharedType(T:KResult)
            where Base = threadId(allocatedDuration) +Int Fresh
		[structural] 

      syntax K ::= "cuda-alloc-shared" "(" Id "," Nat ")"
      rule [cuda-alloc-shared-skip]:
            <k> cuda-alloc-shared(X:Id, _) => . ... </k>
            <bid> BId:Nat </bid>
            <gid> GId:Nat </gid>
            <cudaShared>... 
                  cuda-shared-loc(GId:Nat, BId:Nat, X:Id) |-> _:Nat
            ...</cudaShared>
      rule [cuda-alloc-shared]:
            <k> cuda-alloc-shared(X:Id, Len:Nat) 
                  => alloc(loc(Base:Nat, 0, 0), Len)
                  ~> cuda-set-mdevice(loc(Base:Nat, 0, 0)) 
            ...</k>
            <bid> BId:Nat </bid>
            <gid> GId:Nat </gid>
            <cudaShared>... 
                  . => (cuda-shared-loc(GId:Nat, BId:Nat, X:Id) |-> loc(Base:Nat, 0, 0))
            ...</cudaShared>
            <freshNat> Fresh:Nat => Fresh:Nat +Int 1 </freshNat>
            where Base = threadId(allocatedDuration) +Int Fresh

      syntax K ::= "cuda-set-mdevice" "(" Nat ")"
      rule [cuda-set-mdevice]:
            <k> cuda-set-mdevice(loc(Base:Nat, 0, 0)) => . ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties>... (. => cuda-mdevice) </properties>
            ...</object>

      syntax K ::= "cuda-decl-function-device" "(" Id "," K ")"
      rule [cuda-decl-function-new]:
            <k> cuda-decl-function(X:Id, T:KResult) 
                => cuda-decl-function-device(X:Id, T:KResult) ...</k>
            <cudaGlobalFunctions> Fs:Set (. => SetItem(X:Id)) </cudaGlobalFunctions>
            when isCudaGlobalType(T:KResult)
            andBool notBool (X:Id in Fs:Set)
      rule [cuda-decl-function-old]:
            <k> cuda-decl-function(X:Id, T:KResult) 
                => cuda-decl-function-device(X:Id, T:KResult) ...</k>
            <cudaGlobalFunctions> Fs:Set </cudaGlobalFunctions>
            when isCudaGlobalType(T:KResult)
            andBool X:Id in Fs:Set
      rule [cuda-decl-function-not]:
            <k> cuda-decl-function(X:Id, T:KResult) 
                => cuda-decl-function-device(X:Id, T:KResult) ...</k>
            when notBool isCudaGlobalType(T:KResult)

      rule [cuda-decl-function-device-new]:
            <k> cuda-decl-function-device(X:Id, T:KResult) => . ...</k>
            <cudaDeviceFunctions> Fs:Set (. => SetItem(X:Id)) </cudaDeviceFunctions>
            when isCudaDeviceType(T:KResult)
            andBool notBool (X:Id in Fs:Set)
      rule [cuda-decl-function-device-old]:
            <k> cuda-decl-function-device(X:Id, T:KResult) => . ...</k>
            <cudaDeviceFunctions> Fs:Set </cudaDeviceFunctions>
            when isCudaDeviceType(T:KResult)
            andBool X:Id in Fs:Set
      rule [cuda-decl-function-device-not]:
            <k> cuda-decl-function-device(X:Id, T:KResult) => . ...</k>
            when notBool isCudaDeviceType(T:KResult)

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

      syntax K ::= "cuda-put-in-stream" "(" K "," Nat ")" 

      rule [cuda-spawn2]:
           <k> CudaSpawn2(FCall:K, tv(NBlocks:Nat, _), tv(NThreads:Nat, _))
                 => cuda-check-isglobal(FCall:K) 
                 ~> cuda-launch-kernel(FCall, NBlocks, NThreads, 0, 0)
           ...</k>
           <gid> 0 </gid>
           [structural]
      rule [cuda-spawn3]:
           <k> CudaSpawn3(FCall:K, tv(NBlocks:Nat, _), tv(NThreads:Nat, _), tv(NShared:Nat, _))
                 => cuda-check-isglobal(FCall:K)
                 ~> cuda-launch-kernel(FCall, NBlocks, NThreads, NShared, 0)
           ...</k>
           <gid> 0 </gid>
           [structural]
      rule [cuda-spawn4]:
           <k> CudaSpawn4(FCall:K, tv(NBlocks:Nat, _), tv(NThreads:Nat, _), tv(NShared:Nat, _), tv(SId:Nat, _))
                 => cuda-check-isglobal(FCall:K) 
                 ~> cuda-launch-kernel(FCall:K, NBlocks, NThreads, NShared, SId)
           ...</k>
           <gid> 0 </gid>
           [structural]

      syntax K ::= "cuda-check-isglobal" "(" K ")" 
      rule [cuda-check-isglobal-pass]:
           <k> cuda-check-isglobal(Call(tv(Loc:Nat, _), _)) => . ...</k>
           <functions>... Loc:Nat |-> functionObject(X:Id, _, _) ...</functions>
           <cudaGlobalFunctions>... SetItem(X:Id) ...</cudaGlobalFunctions>
           [structural]
      rule [cuda-check-isglobal-fail]:
           < k => finalComputation > cuda-check-isglobal(Call(tv(Loc:Nat, _), _)) ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90000", "CUDA: Attempting to launch a kernel from a function not marked __global__.") </errorCell>)
           <functions>... Loc:Nat |-> functionObject(X:Id, _, _) ...</functions>
           <cudaGlobalFunctions> Fs:Set </cudaGlobalFunctions>
           when notBool(X:Id in Fs:Set)
           [structural]


      // isdevice == ~ishost
      // Note that these take a pointer and not the whole Call.
      // TODO: clean this up.
      rule [cuda-check-isdevice-pass]:
           <k> cuda-check-isdevice(Loc:Nat) => . ...</k>
           <functions>... Loc:Nat |-> functionObject(X:Id, _, _) ...</functions>
           <cudaGlobalFunctions> GFs:Set </cudaGlobalFunctions>
           <cudaDeviceFunctions> DFs:Set </cudaDeviceFunctions>
           when X:Id in GFs:Set
           orBool X:Id in DFs:Set
           [structural]
      rule [cuda-check-isdevice-prototype-pass]:
           <k> cuda-check-isdevice(Loc:Nat) => . ...</k>
           <functions>... Loc:Nat |-> functionPrototype(X:Id, _) ...</functions>
           <cudaGlobalFunctions> GFs:Set </cudaGlobalFunctions>
           <cudaDeviceFunctions> DFs:Set </cudaDeviceFunctions>
           when X:Id in GFs:Set
           orBool X:Id in DFs:Set
           [structural]
      rule [cuda-check-isdevice-fail]:
           < k => finalComputation > cuda-check-isdevice(Loc:Nat) ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90001", "CUDA: Attempting to call a host function from the device.") </errorCell>)
           <functions>... Loc:Nat |-> functionObject(X:Id, _, _) ...</functions>
           <cudaGlobalFunctions> GFs:Set </cudaGlobalFunctions>
           <cudaDeviceFunctions> DFs:Set </cudaDeviceFunctions>
           when notBool(X:Id in GFs:Set)
           andBool notBool(X:Id in DFs:Set)
           [structural]
      rule [cuda-check-isdevice-prototype-fail]:
           < k => finalComputation > cuda-check-isdevice(Loc:Nat) ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90001", "CUDA: Attempting to call a host function from the device.") </errorCell>)
           <functions>... Loc:Nat |-> functionPrototype(X:Id, _) ...</functions>
           <cudaGlobalFunctions> GFs:Set </cudaGlobalFunctions>
           <cudaDeviceFunctions> DFs:Set </cudaDeviceFunctions>
           when notBool(X:Id in GFs:Set)
           andBool notBool(X:Id in DFs:Set)
           [structural]

      rule [cuda-check-ishost-pass]:
           <k> cuda-check-ishost(Loc:Nat) => . ...</k>
           <functions>... Loc:Nat |-> functionObject(X:Id, _, _) ...</functions>
           <cudaGlobalFunctions> GFs:Set </cudaGlobalFunctions>
           <cudaDeviceFunctions> DFs:Set </cudaDeviceFunctions>
           when notBool(X:Id in GFs:Set)
           andBool notBool(X:Id in DFs:Set)
           [structural]
      rule [cuda-check-ishost-prototype-pass]:
           <k> cuda-check-ishost(Loc:Nat) => . ...</k>
           <functions>... Loc:Nat |-> functionPrototype(X:Id, _) ...</functions>
           <cudaGlobalFunctions> GFs:Set </cudaGlobalFunctions>
           <cudaDeviceFunctions> DFs:Set </cudaDeviceFunctions>
           when notBool(X:Id in GFs:Set)
           andBool notBool(X:Id in DFs:Set)
           [structural]
      rule [cuda-check-ishost-fail]:
           < k => finalComputation > cuda-check-ishost(Loc:Nat) ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90002", "CUDA: Attempting to call a device function from the host.") </errorCell>)
           <functions>... Loc:Nat |-> functionObject(X:Id, _, _) ...</functions>
           <cudaGlobalFunctions> GFs:Set </cudaGlobalFunctions>
           <cudaDeviceFunctions> DFs:Set </cudaDeviceFunctions>
           when X:Id in GFs:Set
           orBool X:Id in DFs:Set
           [structural]
      rule [cuda-check-ishost-prototype-fail]:
           < k => finalComputation > cuda-check-ishost(Loc:Nat) ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90002", "CUDA: Attempting to call a device function from the host.") </errorCell>)
           <functions>... Loc:Nat |-> functionPrototype(X:Id, _) ...</functions>
           <cudaGlobalFunctions> GFs:Set </cudaGlobalFunctions>
           <cudaDeviceFunctions> DFs:Set </cudaDeviceFunctions>
           when X:Id in GFs:Set
           orBool X:Id in DFs:Set
           [structural]

      rule < k => finalComputation > cuda-launch-kernel(_, 0, _, _, _) ...</ k => finalComputation > 
           (.Bag => <errorCell> Error("90003", "CUDA: Kernel launch with NBlocks zero.") </errorCell>)
      rule < k => finalComputation > cuda-launch-kernel(_, _, 0, _, _) ...</ k => finalComputation > 
           (.Bag => <errorCell> Error("90004", "CUDA: Kernel launch with NThreads zero.") </errorCell>)

      // TODO: return value.
      rule <k> cuda-launch-kernel(FCall:K, NBlocks:Nat, NThreads:Nat, NShared:Nat, SId:Nat) 
               => cuda-put-in-stream(cuda-next-grid(GId), SId) ~> cuda-success ...</k>
           <nextGid> GId:Nat => GId +Int 1 </nextGid> 
           <grids>... (. => GId |-> cuda-grid(FCall, NBlocks, NThreads, NShared)) ...</grids>

      rule <streamContents> cuda-next-grid(GId:Nat) =>
            cuda-launch-threads(GId, NBlocks -Int 1, NThreads -Int 1) ...</streamContents>
            <grids>... GId |-> cuda-grid(_, NBlocks:Nat, NThreads:Nat, _) ...</grids>
 
       rule [cuda-launch-thread]:
            <streamContents> cuda-launch-thread(TBody:K, GId:Nat, BId:Nat, TId:Nat) => . ...</streamContents>
            <grids> ... GId |-> cuda-grid(_, _, _, NShared:Nat) ... </grids>
            // TODO: must we clone the main thread?
            <nextThreadId> ThreadId:Nat => ThreadId +Int 1</nextThreadId>
            <threadStatus> Status:Map => Status[threadRunning / ThreadId] </threadStatus>
            <thread>
                  <gid> 0 </gid>
                  <bid> 0 </bid>
                  <tid> 0 </tid>
                  <threadAccess> _:Set </threadAccess>
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
                  <threadAccess> cuda-access-device-read cuda-access-device-write </threadAccess>
                  <nextLoc> loc(threadId(ThreadId) +Int 0, 0, 0) </nextLoc>
                  <threadId> ThreadId </threadId>
                  C:Bag
                  <k> cuda-alloc-shared(Identifier("$dynamic"), NShared:Nat) ~> TBody:K </k>
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
                  (. => cuda-launch-thread((Computation(FCall) ~> cuda-sync-grid ~> cuda-free-shared ~> cuda-join(GId)), GId, 0, 0))
                  ~> (cuda-launch-threads(GId:Nat, 0, 0) => cuda-join(GId))
            ...</streamContents>
            <grids> ... GId |-> cuda-grid(FCall:K, NBlocks:Nat, NThreads:Nat, NShared:Nat) ... </grids>
            [structural]

      /*@ Launch the first thread of a block. */

      rule [cuda-launch-block-head]:
            <streamContents> 
                  (. => cuda-launch-thread((Computation(FCall) ~> cuda-sync-grid ), GId, BId, 0)) 
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

      /* @ Free shared memory. */

      // gid, bid
      syntax K ::= "cuda-shared-loc" "(" Nat "," Nat "," Id ")"

      // gid
      syntax K ::= "cuda-free-shared"

      // TODO: Not bothering for now.
      rule [cuda-free-shared]: cuda-free-shared => .
      // rule [cuda-free-shared]:
      //       <k> (. => deleteBlock(Loc:Nat))
      //             ~> cuda-free-shared ...</k>
      //       <gid> GId:Nat </gid>
      //       <cudaShared>... (cuda-shared-loc(GId:Nat, _, _) |-> Loc:Nat) => . ...</cudaShared>
      // rule [cuda-free-shared-done]:
      //       <k> cuda-free-shared => . ...</k>
      //       <gid> GId:Nat </gid>
      //       <cudaShared> CudaShared:Map </cudaShared>
      //       when notBool(cuda-shared-loc(GId:Nat, _, _) in keys CudaShared:Map)
  
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
            <k> cuda-sync(2, GId:Nat, BId:Nat, SuccTId:Nat, NThreads:Nat) => cuda-success...</k>
            <k> cuda-sync(0 => 2, GId, BId, TId:Nat, NThreads) ...</k>
            when SuccTId ==Int TId +Int 1
      rule <k> cuda-sync(2, _, _, 0, _) => cuda-success ...</k> 

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

      /*@ Destroy stream. */

      rule [cudaStreamDestory-existing]:
            <k> prepareBuiltin((Identifier("cudaStreamDestroy")), tv(SId:Nat, _))
                  => cuda-success
            ...</k>
            <initializedStreams>... SetItem(SId:Nat) => . ...</initializedStreams>
      rule [cudaStreamDestory-non-existing]:
            < k => finalComputation > prepareBuiltin((Identifier("cudaStreamDestroy")), tv(SId:Nat, _))
            ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90400", "CUDA: Attempting to destroy a non-existing stream.") </errorCell>)
            <initializedStreams> Streams:Set </initializedStreams>
            when notBool(SId in Streams)

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

      // TODO: return value (also argument).
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
            < k => finalComputation > cuda-stream-synchronize(SId:Nat) ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90402", "CUDA: Call to cudaStreamSynchronize with an uncreated stream.") </errorCell>)
            <initializedStreams> Streams:Set </initializedStreams>
            when notBool(SId in Streams)

      /*@ Remove defunct streams. */

      rule (<stream>... <sid>SId:Nat</sid> 
            <streamContents> .K </streamContents> ...</stream> => .)
            <activeStreams> ... (SetItem(SId)=>.) ...</activeStreams>

      /*@ Device synchronization. */

      syntax K ::= "cuda-device-synchronize"
      rule <k> prepareBuiltin((Identifier("cudaDeviceSynchronize")), _) => cuda-device-synchronize ...</k>
      rule <k> cuda-device-synchronize => cuda-success ...</k>
           <activeStreams> .Set </activeStreams> // i.e., all streams are empty.

      /*@ \subsection{Memory stuff.} */
      
      syntax BagItem ::= "cuda-mdevice"
                       | "cuda-mshared"
      syntax K ::= "cuda-malloc" "(" Nat ")"
      rule [cudaMalloc]:
            <k> prepareBuiltin((Identifier("cudaMalloc")), (ReturnPtrLoc:KResult,,tv(Len:Nat, T:KResult)))
                  => Computation(*(ReturnPtrLoc) := cuda-malloc(Len:Nat))
                  ~> cuda-success
            ...</k>
      rule [cuda-malloc]:
            <k> cuda-malloc(Len:Nat) 
                  => tv(loc(Base, 0, 0), t(.Set, pointerType(t(.Set, void))))
            ...</k>
            <malloced>... .Map => loc(Base, 0, 0) |-> Len:Nat ...</malloced>
            <freshNat> Fresh:Nat => Fresh:Nat +Int 1 </freshNat>
            <memory>...
                  (.Bag => <object>...
                        <basePtr> Base:Nat </basePtr>
                        <oLength> Len:Nat </oLength>
                        <properties> cuda-mdevice </properties>
                  ...</object>)
            ...</memory>
            where Base = threadId(allocatedDuration) +Int Fresh

      // Just like regular free.
      rule [cudaFree]:
            <k> prepareBuiltin((Identifier("cudaFree")), tv(Loc:Nat, t(_, pointerType(_))))
                  => deleteSizedBlock(Loc:Nat, Len:Nat) 
                  ~> cuda-success
            ...</k>
            <malloced>... Loc:Nat |-> Len:Nat => .Map ...</malloced>

      // object properties, thread access
      syntax K ::= "cuda-mem-check" "(" SetItem "," Set ")"
      // TODO: probably not right.
      rule [cuda-read-ignore-unmalloced]:
            <k> cuda-read-check(loc(N:Nat +Int _:Nat, _, _)) => . ...</k>
            when N:Nat =/=K threadId(allocatedDuration) // Ignore everything but malloc'd.
      rule [cuda-read-check-device]:
            <k> cuda-read-check(loc(Base:Nat, _, _)) => cuda-mem-check(cuda-access-device-read, Access:Set) ...</k>
            <threadAccess> Access:Set </threadAccess>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when cuda-mdevice in Attr:Bag
      rule [cuda-read-check-host]:
            <k> cuda-read-check(loc(Base:Nat, _, _)) => cuda-mem-check(cuda-access-host-read, Access:Set) ...</k>
            <threadAccess> Access:Set </threadAccess>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when notBool(cuda-mdevice in Attr:Bag)

      rule [cuda-write-ignore-unmalloced]:
            <k> cuda-write-check(loc(N:Nat +Int _:Nat, _, _)) => . ...</k>
            when N:Nat =/=K threadId(allocatedDuration) // Ignore everything but malloc'd.
      rule [cuda-write-check-device]:
            <k> cuda-write-check(loc(Base:Nat, _, _)) => cuda-mem-check(cuda-access-device-write, Access:Set) ...</k>
            <threadAccess> Access:Set </threadAccess>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when cuda-mdevice in Attr:Bag
      rule [cuda-write-check-host]:
            <k> cuda-write-check(loc(Base:Nat, _, _)) => cuda-mem-check(cuda-access-host-write, Access:Set) ...</k>
            <threadAccess> Access:Set </threadAccess>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when notBool(cuda-mdevice in Attr:Bag)

      rule [cuda-mem-check-pass]:
            <k> cuda-mem-check(Request:SetItem, Access:Set) => . ...</k>
            when Request in Access
      rule [cuda-mem-check-read-device-fail]:
            < k => finalComputation > cuda-mem-check(cuda-access-device-read, Access:Set) ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90500", "CUDA: Device memory access from host (read).") </errorCell>)
            when notBool(cuda-access-device-read in Access)
      rule [cuda-mem-check-read-host-fail]:
            < k => finalComputation > cuda-mem-check(cuda-access-host-read, Access:Set) ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90501", "CUDA: Host memory access from device (read).") </errorCell>)
            when notBool(cuda-access-host-read in Access)
      rule [cuda-mem-check-write-device-fail]:
            < k => finalComputation > cuda-mem-check(cuda-access-device-write, Access:Set) ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90502", "CUDA: Device memory access from host (write).") </errorCell>)
            when notBool(cuda-access-device-write in Access)
      rule [cuda-mem-check-write-host-fail]:
            < k => finalComputation > cuda-mem-check(cuda-access-host-write, Access:Set) ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90503", "CUDA: Host memory access from device (write).") </errorCell>)
            when notBool(cuda-access-host-write in Access)

      rule [host-thread-access]:
            <gid> 0 </gid>
            <threadAccess> .Set => (cuda-access-host-read cuda-access-host-write) </threadAccess>

      ops cudaMemcpyHostToDevice cudaMemcpyDeviceToHost : -> Nat
      macro cudaMemcpyHostToDevice = 1
      macro cudaMemcpyDeviceToHost = 2

      //                                 dst,    src,    count
      syntax K ::= "cuda-memcpy-h2d" "(" Nat "," Nat "," Nat ")"
      syntax K ::= "cuda-memcpy-d2h" "(" Nat "," Nat "," Nat ")"
      syntax K ::= "cuda-memcpy" "(" Nat "," Nat "," Nat ")"
      rule [cuda-memcpy-h2d]:
            <k> prepareBuiltin((Identifier("cudaMemcpy")), (tv(Dst:Nat, t(_, pointerType(_))),, tv(Src:Nat, t(_, pointerType(_))),, tv(Count:Nat, _),, tv(Kind:Nat, _)))
                  => cuda-put-in-stream(cuda-memcpy-h2d(Dst:Nat, Src:Nat, Count:Nat), 0) 
                  ~> cuda-stream-synchronize(0)
            ...</k>
            when Kind:Nat ==Int cudaMemcpyHostToDevice
            [structural]
      rule [cuda-memcpy-d2h]:
            <k> prepareBuiltin((Identifier("cudaMemcpy")), (tv(Dst:Nat, t(_, pointerType(_))),, tv(Src:Nat, t(_, pointerType(_))),, tv(Count:Nat, _),, tv(Kind:Nat, _)))
                  => cuda-put-in-stream(cuda-memcpy-d2h(Dst:Nat, Src:Nat, Count:Nat), 0) 
                  ~> cuda-stream-synchronize(0)
            ...</k>
            when Kind:Nat ==Int cudaMemcpyDeviceToHost
            [structural]
      rule [cuda-memcpy-async-h2d]:
            <k> prepareBuiltin((Identifier("cudaMemcpyAsync")), (tv(Dst:Nat, t(_, pointerType(_))),, tv(Src:Nat, t(_, pointerType(_))),, tv(Count:Nat, _),, tv(Kind:Nat, _),, tv(SId:Nat, _)))
                  => cuda-put-in-stream(cuda-memcpy-h2d(Dst:Nat, Src:Nat, Count:Nat), SId) 
                  ~> cuda-success
            ...</k>
            when Kind:Nat ==Int cudaMemcpyHostToDevice
            [structural]
      rule [cuda-memcpy-async-d2h]:
            <k> prepareBuiltin((Identifier("cudaMemcpyAsync")), (tv(Dst:Nat, t(_, pointerType(_))),, tv(Src:Nat, t(_, pointerType(_))),, tv(Count:Nat, _),, tv(Kind:Nat, _),, tv(SId:Nat, _)))
                  => cuda-put-in-stream(cuda-memcpy-d2h(Dst:Nat, Src:Nat, Count:Nat), SId) 
                  ~> cuda-success
            ...</k>
            when Kind:Nat ==Int cudaMemcpyDeviceToHost
            [structural]
      
      // Using a single thread for memory transfers might model the
      // architecture better.
      syntax SetItem ::= "cuda-access-device-read"
                       | "cuda-access-device-write"
                       | "cuda-access-host-read"
                       | "cuda-access-host-write"
      rule [cuda-memcpy-h2d-stream]:
            <streamContents> cuda-memcpy-h2d(Dst:Nat, Src:Nat, Count:Nat) => cuda-join(GId:Nat) ...</streamContents>
            <nextGid> GId:Nat => GId:Nat +Int 1 </nextGid>
            <thread> <gid> 0 </gid> <k> _:K </k> C:Bag </thread> // TODO: probably grabbing too much.
            (.Bag =>
            <thread>
                  <k> cuda-memcpy(Dst:Nat, Src:Nat, Count:Nat) ~> cuda-join(GId:Nat) </k>
                  <gid> GId:Nat </gid>
                  <threadAccess> cuda-access-host-read cuda-access-device-write </threadAccess>
                  C:Bag
            </thread>)
      rule [cuda-memcpy-d2h-stream]:
            <streamContents> cuda-memcpy-d2h(Dst:Nat, Src:Nat, Count:Nat) => cuda-join(GId:Nat) ...</streamContents>
            <nextGid> GId:Nat => GId:Nat +Int 1 </nextGid>
            <thread> <gid> 0 </gid> <k> _:K </k> <threadAccess> _:Set </threadAccess> C:Bag </thread> // TODO: probably grabbing too much.
            (.Bag =>
            <thread>
                  <k> cuda-memcpy(Dst:Nat, Src:Nat, Count:Nat) ~> cuda-join(GId:Nat) </k>
                  <gid> GId:Nat </gid>
                  <threadAccess> cuda-access-device-read cuda-access-host-write </threadAccess>
                  C:Bag
            </thread>)

      rule [cuda-memcpy-read]:
            <k> (.K => read(Src:Nat, t(.Set, char)))
                  ~> cuda-memcpy(_, (Src:Nat => Src:Nat +Int 1), (Count:Nat => Count:Nat -Int 1))
            ...</k>
            when Count:Nat =/=Int 0
            [structural]
      
      rule [cuda-memcpy-write]:
            <k> (tv(I:Int, T:KResult) => write(lv(Dst:Nat, t(.Set, char)), tv(I:Int, T:KResult)))
                  ~> cuda-memcpy((Dst:Nat => Dst:Nat +Int 1), _, _)
            ...</k>
            [structural]

      rule [cuda-memcpy-done]:
            <k> cuda-memcpy(_, _, 0) => . ...</k>
            [structural]

end module

