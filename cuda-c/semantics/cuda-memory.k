load dynamic-c-semantics
load cuda-helpers
load cuda-streams

module CUDA-MEMORY is
      including DYNAMIC-C-SEMANTICS
      including CUDA-HELPERS
      including CUDA-STREAMS

      syntax K ::= "cuda-set-mdevice" "(" Nat ")"
      rule [cuda-set-mdevice]:
            <k> cuda-set-mdevice(loc(Base:Nat, 0, 0)) => . ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties>... (. => cuda-mdevice) </properties>
            ...</object>

      syntax K ::= "cuda-set-mhost" "(" Nat ")"
      rule [cuda-set-mhost]:
            <k> cuda-set-mhost(loc(Base:Nat, 0, 0)) => . ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties>... (. => cuda-mhost) </properties>
            ...</object>

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
  

      /*@ \subsection{Memory stuff.} */
      
      syntax BagItem ::= "cuda-mdevice"
                       | "cuda-mshared"
                       | "cuda-mhost"
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
            <streamContents> cuda-memset(Dst:Nat, Val:KResult, Count:Nat) => cuda-join(GId:Nat) ...</streamContents>
            <nextGid> GId:Nat => GId:Nat +Int 1 </nextGid>
            <thread> <gid> 0 </gid> <k> _:K </k> <threadAccess> _:Set </threadAccess> C:Bag </thread> // TODO: probably grabbing too much.
            (.Bag =>
            <thread>
                  <k> cuda-memset(Dst:Nat, Val:KResult, Count:Nat) ~> cuda-join(GId:Nat) </k>
                  <gid> GId:Nat </gid>
                  <threadAccess> cuda-access-device-write </threadAccess>
                  C:Bag
            </thread>)

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
                  => .
            ...</k>
      rule [threadfence]:
            <k> prepareBuiltin((Identifier("__threadfence")), _)
                  => .
            ...</k>
      rule [threadfence-system]:
            <k> prepareBuiltin((Identifier("__threadfence_system")), _)
                  => .
            ...</k>
end module
