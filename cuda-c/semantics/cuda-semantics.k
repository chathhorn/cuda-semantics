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

      ops cuda-success : -> Nat
      macro cuda-success = tv(0, t(.Set, int))

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

      rule < k => finalComputation > cuda-launch-kernel(_, 0, _, _, _) ...</ k => finalComputation > 
           (.Bag => <errorCell> Error("90000", "CUDA: Kernel launch with NBlocks zero.") </errorCell>)
      rule < k => finalComputation > cuda-launch-kernel(_, _, 0, _, _) ...</ k => finalComputation > 
           (.Bag => <errorCell> Error("90001", "CUDA: Kernel launch with NThreads zero.") </errorCell>)

      rule [cuda-put-in-existing-stream]:
           <k> cuda-put-in-stream(Contents:K, SId:Nat) => . ...</k>
           <stream>... 
                 <sid> SId </sid> 
                 <streamContents> ... (. => Contents) </streamContents> 
            ...</stream>
           [structural]
      rule [cuda-put-in-new-stream]:
           <k> cuda-put-in-stream(Contents:K, SId:Nat) => . ...</k>
           <streams> Streams:Set (. => SetItem(SId)) </streams>
           (. => <stream>... <sid> SId </sid> 
                 <streamContents> Contents </streamContents> 
            ...</stream>)
           when notBool(SId in Streams)
           [structural]

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

      /*@ Device synchronization. */
      syntax K ::= "cuda-device-synchronize"
      rule <k> prepareBuiltin((Identifier("cudaDeviceSynchronize")), _) => cuda-device-synchronize ...</k>
      rule <k> cuda-device-synchronize => cuda-success ...</k>
           <streams> .Set </streams> // i.e., all streams are empty.

      /*@ Stream synchronization. */
      // TODO: return value (also argument).
      syntax K ::= "cuda-stream-synchronize" "(" Nat ")"
      rule <k> prepareBuiltin((Identifier("cudaStreamSynchronize")), tv(SId:Nat, t(_, int))) 
                  => cuda-stream-synchronize(SId:Nat) ...</k>
      rule <k> cuda-stream-synchronize(SId:Nat) => cuda-success ...</k>
            <streams> Streams:Set </streams>
            when notBool(SId in Streams)

      /*@ Remove defunct streams. */
      rule (<stream>... <sid>SId:Nat</sid> 
            <streamContents> .K </streamContents> ...</stream> => .)
            <streams> ... (SetItem(SId)=>.) ...</streams>

      /*@ \subsection{Memory stuff.} */
      
      syntax BagItem ::= "mdevice"
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
				<properties> mdevice </properties>
			...</object>)
		...</memory>
		where Base = threadId(allocatedDuration) +Int Fresh

      // Just like regular free.
	rule [cudaFree]:
		<k> prepareBuiltin((Identifier("cudaFree")), tv(Loc:Nat, t(_, pointerType(_))))
			=> deleteSizedBlock(Loc:Nat, Len:Nat) 
			~> skipval
		...</k>
		<malloced>... Loc:Nat |-> Len:Nat => .Map ...</malloced>


      syntax K ::= "cuda-mem-check" "(" K ")"

      rule [cuda-read-check]:
            <k> cuda-read-check(Loc:Nat) 
            => cuda-mem-check(Loc:Nat)
            ...</k>

      rule [cuda-write-check]:
            <k> cuda-write-check(Loc:Nat) 
            => cuda-mem-check(Loc:Nat)
            ...</k>

      // This disables check in the memcpy thread.
      // TODO: I should make sure the h2d/d2h distinction checks out.
	rule [cuda-mem-check-memcpy-pass]:
		<k> cuda-mem-check(loc(Base:Nat, Offset:Nat, 0)) => . ...</k>
            <gid> 1 </gid>

      // TODO: CUDA: Buffer.
	rule [cuda-mem-check-device-pass]:
		<k> cuda-mem-check(loc(Base:Nat, Offset:Nat, 0)) => .
		...</k>
            <gid> GId:Nat </gid>
		<object>...
			<basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
		...</object>
            when GId >Int 1 // 1 is the "memcpy" thread.
		andBool mdevice in Attr:Bag
		[large structural]
	
	rule [cuda-mem-check-device-fail]:
		< k => finalComputation > cuda-mem-check(loc(Base:Nat, Offset:Nat, 0)) 
            ...</k => finalComputation >
           (.Bag => <errorCell> Error("90500", "CUDA: Host memory access from device.") </errorCell>)
            <gid> GId:Nat </gid>
		<object>...
			<basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
		...</object>
            when GId >Int 1
		andBool notBool(mdevice in Attr:Bag)
		[large structural]
	
	rule [cuda-mem-check-host-pass]:
		<k> cuda-mem-check(loc(Base:Nat, Offset:Nat, 0)) => .
		...</k>
            <gid> 0 </gid>
		<object>...
			<basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
		...</object>
		when notBool(mdevice in Attr:Bag)
		[large structural]
	
	rule [cuda-mem-check-host-fail]:
		< k => finalComputation > cuda-mem-check(loc(Base:Nat, Offset:Nat, 0)) 
            ...</k => finalComputation >
           (.Bag => <errorCell> Error("90501", "CUDA: Device memory access from host.") </errorCell>)
            <gid> 0 </gid>
		<object>...
			<basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
		...</object>
		when mdevice in Attr:Bag
		[large structural]
	
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
                  ~> cuda-device-synchronize
		...</k>
            where Kind:Nat = cudaMemcpyHostToDevice
		[structural]
	rule [cuda-memcpy-d2h]:
		<k> prepareBuiltin((Identifier("cudaMemcpy")), (tv(Dst:Nat, t(_, pointerType(_))),, tv(Src:Nat, t(_, pointerType(_))),, tv(Count:Nat, _),, tv(Kind:Nat, _)))
			=> cuda-put-in-stream(cuda-memcpy-d2h(Dst:Nat, Src:Nat, Count:Nat), 0) 
                  ~> cuda-device-synchronize
		...</k>
            where Kind:Nat = cudaMemcpyDeviceToHost
		[structural]
	rule [cuda-memcpy-async-h2d]:
		<k> prepareBuiltin((Identifier("cudaMemcpyAsync")), (tv(Dst:Nat, t(_, pointerType(_))),, tv(Src:Nat, t(_, pointerType(_))),, tv(Count:Nat, _),, tv(Kind:Nat, _),, tv(SId:Nat, _)))
			=> cuda-put-in-stream(cuda-memcpy-h2d(Dst:Nat, Src:Nat, Count:Nat), SId) 
                  ~> cuda-success
		...</k>
            where Kind:Nat = cudaMemcpyHostToDevice
		[structural]
	rule [cuda-memcpy-async-d2h]:
		<k> prepareBuiltin((Identifier("cudaMemcpyAsync")), (tv(Dst:Nat, t(_, pointerType(_))),, tv(Src:Nat, t(_, pointerType(_))),, tv(Count:Nat, _),, tv(Kind:Nat, _),, tv(SId:Nat, _)))
			=> cuda-put-in-stream(cuda-memcpy-d2h(Dst:Nat, Src:Nat, Count:Nat), SId) 
                  ~> cuda-success
		...</k>
            where Kind:Nat = cudaMemcpyDeviceToHost
		[structural]
      
      // TODO: I don't think cuda-join(1) is enough.
      // TODO: don't ellide d2h/h2d here so we can enforce the host/device
      // boundary better.
      rule [cuda-memcpy-h2d-stream]:
            <streamContents> cuda-memcpy-h2d(Dst:Nat, Src:Nat, Count:Nat) => cuda-join(1) ...</streamContents>
            <thread> <gid> 0 </gid> <k> _:K </k> C:Bag </thread> // TODO: probably grabbing too much.
            (.Bag =>
            <thread>
                  <gid> 1 </gid>
                  <k> cuda-memcpy(Dst:Nat, Src:Nat, Count:Nat) ~> cuda-join(1) </k>
                  C:Bag
            </thread>)
      rule [cuda-memcpy-d2h-stream]:
            <streamContents> cuda-memcpy-d2h(Dst:Nat, Src:Nat, Count:Nat) => cuda-join(1) ...</streamContents>
            <thread> <gid> 0 </gid> <k> _:K </k> C:Bag </thread> // TODO: probably grabbing too much.
            (.Bag =>
            <thread>
                  <gid> 1 </gid>
                  <k> cuda-memcpy(Dst:Nat, Src:Nat, Count:Nat) ~> cuda-join(1) </k>
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


