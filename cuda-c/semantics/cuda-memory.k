load dynamic-c-semantics
load cuda-helpers
load cuda-headers

module CUDA-MEMORY is
      including CUDA-MEMORY-HEADER
      including CUDA-THREADS-HEADER
      including CUDA-STREAMS-HEADER
      including CUDA-DEVICES-HEADER
      including CUDA-HELPERS
      including DYNAMIC-C-SEMANTICS

      rule [cuda-set-mdevice]:
            <k> cuda-set-mdevice(loc(Base:Nat, 0, 0)) => . ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties>... (. => cuda-mdevice) </properties>
            ...</object>

      rule [cuda-set-mhost]:
            <k> cuda-set-mhost(loc(Base:Nat, 0, 0)) => . ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties>... (. => cuda-mhost) </properties>
            ...</object>

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

      /* @ Free shared memory. */

      // gid

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
  

      /*@ \subsection{Memory stuff.} */
      
      syntax K ::= "cuda-malloc" "(" Nat ")"
                 | "cuda-host-alloc" "(" Nat ")"
      rule [cudaMalloc]:
            <k> prepareBuiltin((Identifier("cudaMalloc")), (ReturnPtrLoc:KResult,,tv(Len:Nat, T:KResult)))
                  => Computation(*(ReturnPtrLoc) := cuda-malloc(Len:Nat))
                  ~> cuda-success
            ...</k>
      rule [cuda-malloc]:
            <k> cuda-malloc(Len:Nat) 
                  => alloc(Loc, Len:Nat)
                  ~> cuda-set-mdevice(Loc)
                  ~> tv(Loc, t(.Set, pointerType(t(.Set, void))))
            ...</k>
            <malloced>... .Map => Loc |-> Len:Nat ...</malloced>
            <freshNat> Fresh:Nat => Fresh:Nat +Int 1 </freshNat>
		where Loc = loc(threadId(allocatedDuration) +Int Fresh:Nat, 0, 0)
		[structural]

      rule [cudaMallocHost]:
            <k> prepareBuiltin((Identifier("cudaMallocHost")), (ReturnPtrLoc:KResult,,tv(Len:Nat, T:KResult)))
                  => Computation(*(ReturnPtrLoc) := cuda-host-alloc(Len:Nat))
                  ~> cuda-success
            ...</k>

      rule [cudaHostAlloc]:
            <k> prepareBuiltin((Identifier("cudaHostAlloc")), (ReturnPtrLoc:KResult,,tv(Len:Nat, T:KResult),,tv(Flags:Nat, _)))
                  => Computation(*(ReturnPtrLoc) := cuda-host-alloc(Len:Nat))
                  ~> cuda-success
            ...</k>
      rule [cuda-host-alloc]:
            <k> cuda-host-alloc(Len:Nat) 
                  => alloc(Loc, Len:Nat)
                  ~> cuda-set-mdevice(Loc)
                  ~> cuda-set-mhost(Loc)
                  ~> tv(Loc, t(.Set, pointerType(t(.Set, void))))
            ...</k>
            <malloced>... .Map => Loc |-> Len:Nat ...</malloced>
            <freshNat> Fresh:Nat => Fresh:Nat +Int 1 </freshNat>
		where Loc = loc(threadId(allocatedDuration) +Int Fresh:Nat, 0, 0)
		[structural]

      // Just like regular free.
      rule [cudaFree]:
            <k> prepareBuiltin((Identifier("cudaFree")), tv(Loc:Nat, t(_, pointerType(_))))
                  => deleteSizedBlock(Loc:Nat, Len:Nat) 
                  ~> cuda-success
            ...</k>
            <malloced>... Loc:Nat |-> Len:Nat => .Map ...</malloced>

      rule [cudaFreeHost]:
            <k> prepareBuiltin((Identifier("cudaFreeHost")), tv(Loc:Nat, t(_, pointerType(_))))
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
      rule [cuda-read-ignore-pinned]:
            <k> cuda-read-check(loc(Base:Nat, _, _)) => . ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when cuda-mdevice in Attr:Bag
            andBool cuda-mhost in Attr:Bag
      rule [cuda-read-check-device]:
            <k> cuda-read-check(loc(Base:Nat, _, _)) => cuda-mem-check(cuda-access-device-read, Access:Set) ...</k>
            <threadAccess> Access:Set </threadAccess>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when cuda-mdevice in Attr:Bag
            andBool notBool cuda-mhost in Attr:Bag
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
      rule [cuda-write-ignore-pinned]:
            <k> cuda-write-check(loc(Base:Nat, _, _)) => . ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when cuda-mdevice in Attr:Bag
            andBool cuda-mhost in Attr:Bag
      rule [cuda-write-check-device]:
            <k> cuda-write-check(loc(Base:Nat, _, _)) => cuda-mem-check(cuda-access-device-write, Access:Set) ...</k>
            <threadAccess> Access:Set </threadAccess>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when cuda-mdevice in Attr:Bag
            andBool notBool cuda-mhost in Attr:Bag
      rule [cuda-write-check-host]:
            <k> cuda-write-check(loc(Base:Nat, _, _)) => cuda-mem-check(cuda-access-host-write, Access:Set) ...</k>
            <threadAccess> Access:Set </threadAccess>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when notBool(cuda-mdevice in Attr:Bag)

      // TODO: cuda-mhost && cuda-mdevice (accessible from host and device)

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
                  ~> cuda-device-synchronize
            ...</k>
            when Kind:Nat ==Int cudaMemcpyHostToDevice
            [structural]
      rule [cuda-memcpy-d2h]:
            <k> prepareBuiltin((Identifier("cudaMemcpy")), (tv(Dst:Nat, t(_, pointerType(_))),, tv(Src:Nat, t(_, pointerType(_))),, tv(Count:Nat, _),, tv(Kind:Nat, _)))
                  => cuda-put-in-stream(cuda-memcpy-d2h(Dst:Nat, Src:Nat, Count:Nat), 0) 
                  ~> cuda-device-synchronize
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

      rule [cuda-memcpy-h2d-stream]:
            <streamContents> cuda-memcpy-h2d(Dst:Nat, Src:Nat, Count:Nat)
                  => cuda-launch-thread(
                        (cuda-memcpy(Dst:Nat, Src:Nat, Count:Nat) ~> cuda-join(GId:Nat)),
                        GId:Nat, 0, 0,
                        cuda-access-device-write cuda-access-host-read)
                  ~> cuda-join(GId:Nat) 
            ...</streamContents>
            <nextGid> GId:Nat => GId:Nat +Int 1 </nextGid>
      rule [cuda-memcpy-d2h-stream]:
            <streamContents> cuda-memcpy-d2h(Dst:Nat, Src:Nat, Count:Nat)
                  => cuda-launch-thread(
                        (cuda-memcpy(Dst:Nat, Src:Nat, Count:Nat) ~> cuda-join(GId:Nat)),
                        GId:Nat, 0, 0,
                        cuda-access-device-read cuda-access-host-write)
                  ~> cuda-join(GId:Nat) 
            ...</streamContents>
            <nextGid> GId:Nat => GId:Nat +Int 1 </nextGid>

      // TODO: readActual and write-specific: these bypass the check for
      // reading/writing thread-local memory (e.g., stack), but they also
      // bypass the cuda memory checks.
      rule [cuda-memcpy-read]:
            <k> (.K 
                        => cuda-read-check(Src:Nat)
                        ~> readActual(Src:Nat, t(.Set, char), value(bitSizeofType(t(.Set, char)))))
                  ~> cuda-memcpy(_, (Src:Nat => Src:Nat +Int 1), (Count:Nat => Count:Nat -Int 1))
            ...</k>
            when Count:Nat =/=Int 0
            [computational ndlocal]
      
      rule [cuda-memcpy-write]:
            <k> (tv(I:Int, T:KResult) 
                        => cuda-write-check(Dst:Nat)
                        ~> write-specific(Dst:Nat, tv(I:Int, T:KResult), value(bitSizeofType(T:KResult))))
                  ~> cuda-memcpy((Dst:Nat => Dst:Nat +Int 1), _, _)
            ...</k>
            [computational ndlocal]

      rule [cuda-memcpy-done]:
            <k> cuda-memcpy(_, _, 0) => . ...</k>
            [structural]

      /*@ Memset. */

      //                             dst,    val,        count
      syntax K ::= "cuda-memset" "(" Nat "," KResult "," Nat ")"
      rule [cudaMemset]:
            <k> prepareBuiltin((Identifier("cudaMemset")), (tv(Dst:Nat, t(_, pointerType(_))),, Val:KResult,, tv(Count:Nat, _)))
                  => cuda-put-in-stream(cuda-memset(Dst:Nat, Val:KResult, Count:Nat), 0) 
                  ~> cuda-success
            ...</k>
            [structural]

      rule [cudaMemsetAsync]:
            <k> prepareBuiltin((Identifier("cudaMemsetAsync")), (tv(Dst:Nat, t(_, pointerType(_))),, Val:KResult,, tv(Count:Nat, _),, tv(SId:Nat, _)))
                  => cuda-put-in-stream(cuda-memset(Dst:Nat, Val:KResult, Count:Nat), SId:Nat) 
                  ~> cuda-success
            ...</k>
            [structural]
      
      rule [cuda-memset]:
            <streamContents> cuda-memset(Dst:Nat, Val:KResult, Count:Nat) 
                  => cuda-launch-thread(
                        (cuda-memset(Dst:Nat, Val:KResult, Count:Nat) ~> cuda-join(GId:Nat)),
                        GId:Nat, 0, 0,
                        cuda-access-device-write)
                  ~> cuda-join(GId:Nat) 
            ...</streamContents>
            <nextGid> GId:Nat => GId:Nat +Int 1 </nextGid>

      rule [cuda-memset-write]:
            <k> (.K => write(lv(Dst:Nat, t(.Set, char)), tv(Val:Int, t(.Set, char))))
                  ~> cuda-memset((Dst:Nat => Dst:Nat +Int 1), tv(Val:Int, t(.Set, int)), (Count:Nat => Count:Nat -Int 1))
            ...</k>
            [structural]

      rule [cuda-memset-done]:
            <k> cuda-memset(_, _, 0) => . ...</k>
            [structural]

      /*@ Memory Fence Functions */

      // Nothing to do with these right now.
      rule [threadfence-block]:
            <k> prepareBuiltin((Identifier("__threadfence_block")), _)
                  => skipval
            ...</k>
      rule [threadfence]:
            <k> prepareBuiltin((Identifier("__threadfence")), _)
                  => skipval
            ...</k>
      rule [threadfence-system]:
            <k> prepareBuiltin((Identifier("__threadfence_system")), _)
                  => skipval
            ...</k>
end module
