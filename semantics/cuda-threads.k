load dynamic-c-semantics
load cuda-headers

module CUDA-THREADS is
      including CUDA-DATATYPES-HEADER
      including CUDA-THREADS-HEADER
      including CUDA-DECLARATIONS-HEADER
      including CUDA-MEMORY-HEADER
      including CUDA-STREAMS-HEADER
      including CUDA-RACECHECK-HEADER
      including DYNAMIC-C-SEMANTICS
      including CUDA-PROPERTIES-HEADER
      including CUDA-ERRORS-HEADER

      // call, nblocks, threads per block, nshared, stream
      syntax K ::= "cuda-launch-kernel" "(" K "," K "," K "," Nat "," Nat ")" [strict(2 3)]
                  // grid id, blocks left to spawn, threads left to spawn.
                  | "cuda-launch-threads" "(" Nat "," Nat "," Nat ")"
                  | "cuda-launch-kernel-thread" "(" K "," Nat "," Nat "," Nat ")"
                  // grid id
                  | "cuda-next-grid" "(" Nat ")"
                  // call, nblocks, nthreads, nshared
                  | "cuda-sync-grid"
                  //          token, grid id, block id, this id, total ids, predicate
                  | "cuda-sync" "(" Nat "," Nat "," Nat "," Nat "," Nat "," Nat ")"


      syntax KResult ::= "cuda-dim3" "(" Nat "," Nat "," Nat ")"

      syntax K ::= "cuda-get-dim3" "(" KResult ")"
      rule [cuda-get-dim3-struct]:
            <k> cuda-get-dim3(tv(V:List{K}, T:KResult)) 
                 => cuda-dim3-aux(tv(V:List{K}, T:KResult) . Identifier("x"), 
                                  tv(V:List{K}, T:KResult) . Identifier("y"), 
                                  tv(V:List{K}, T:KResult) . Identifier("z"))
            ...</k>
            when T:KResult ==K unqualifyType(cuda-dim3-type)
      rule [cuda-get-dim3-int]:
            <k> cuda-get-dim3(tv(N:Nat, T:KResult)) 
                 => cuda-dim3(N:Nat, 1, 1)
            ...</k>
            when hasIntegerType(T:KResult)

      syntax K ::= "cuda-dim3-aux" "(" KResult "," KResult "," KResult ")" [strict]
      rule cuda-dim3-aux(tv(V0:Nat, T:KResult), tv(V1:Nat, T:KResult), tv(V2:Nat, T:KResult))
            => cuda-dim3(V0:Nat, V1:Nat, V2:Nat)
            when hasIntegerType(T:KResult)

      rule [cuda-spawn2]:
           <k> CudaSpawn2(FCall:K, Grid:KResult, Block:KResult)
                 => cuda-check-isglobal(FCall:K) 
                 ~> cuda-launch-kernel(FCall:K, cuda-get-dim3(Grid:KResult), cuda-get-dim3(Block:KResult), 0, 0)
           ...</k>
           <gid> 0 </gid>
           [structural]
      rule [cuda-spawn3]:
           <k> CudaSpawn3(FCall:K, Grid:KResult, Block:KResult, tv(NShared:Nat, _))
                 => cuda-check-isglobal(FCall:K)
                 ~> cuda-launch-kernel(FCall:K, cuda-get-dim3(Grid:KResult), cuda-get-dim3(Block:KResult), NShared:Nat, 0)
           ...</k>
           <gid> 0 </gid>
           [structural]
      rule [cuda-spawn4]:
           <k> CudaSpawn4(FCall:K, Grid:KResult, Block:KResult, tv(NShared:Nat, _), tv(SId:Nat, _))
                 => cuda-check-isglobal(FCall:K) 
                 ~> cuda-launch-kernel(FCall:K, cuda-get-dim3(Grid:KResult), cuda-get-dim3(Block:KResult), NShared:Nat, SId:Nat)
           ...</k>
           <gid> 0 </gid>
           [structural]
      
      rule < k => finalComputation > cuda-launch-kernel(_, cuda-dim3(X:Nat, Y:Nat, Z:Nat), _, _, _) ...</ k => finalComputation > 
           (.Bag => <errorCell> Error("90003", "CUDA: Kernel launch with NBlocks zero.") </errorCell>)
           when (X:Nat *Int Y:Nat *Int Z:Nat) ==Int 0
      rule < k => finalComputation > cuda-launch-kernel(_, _, cuda-dim3(X:Nat, Y:Nat, Z:Nat), _, _) ...</ k => finalComputation > 
           (.Bag => <errorCell> Error("90004", "CUDA: Kernel launch with NThreads zero.") </errorCell>)
           when (X:Nat *Int Y:Nat *Int Z:Nat) ==Int 0

      // This is used to store data about the grid and block dimensions in the
      // <cudaGrid/BlockDim> cell. The first parameter is the location of an
      // allocated dim3 struct, the second a cuda-dim3() for easy access to the
      // dimensions.
      syntax KResult ::= "cuda-dim-info" "(" K "," KResult ")"

      syntax K ::= "cuda-put-gridDim" "(" Nat "," KResult ")"
                 | "cuda-put-blockDim" "(" Nat "," KResult ")"

      context cuda-put-gridDim(_, cuda-dim-info(HOLE, _))
      context cuda-put-blockDim(_, cuda-dim-info(HOLE, _))

      rule [cuda-put-gridDim]:
            <k> cuda-put-gridDim(GId:Nat, GridDimInfo:KResult) => . ...</k>
            <cudaGridDim> GridDims:Map => GridDims:Map[GridDimInfo:KResult / GId:Nat] </cudaGridDim>
      rule [cuda-put-blockDim]:
            <k> cuda-put-blockDim(GId:Nat, BlockDimInfo:KResult) => . ...</k>
            <cudaBlockDim> BlockDims:Map => BlockDims:Map[BlockDimInfo:KResult / GId:Nat] </cudaBlockDim>

      rule <k> cuda-launch-kernel(FCall:K, cuda-dim3(GridX:Nat, GridY:Nat, _), cuda-dim3(BlockX:Nat, BlockY:Nat, BlockZ:Nat), NShared:Nat, SId:Nat) 
               => cuda-put-gridDim(GId:Nat, cuda-dim-info(cuda-new-dim3(cuda-dim3(GridX:Nat, GridY:Nat, 1)), cuda-dim3(GridX:Nat, GridY:Nat, 1)))
               ~> cuda-put-blockDim(GId:Nat, cuda-dim-info(cuda-new-dim3(cuda-dim3(BlockX:Nat, BlockY:Nat, BlockZ:Nat)), cuda-dim3(BlockX:Nat, BlockY:Nat, BlockZ:Nat)))
               ~> cuda-put-in-stream(cuda-next-grid(GId), SId) ~> cuda-return-success ...</k>
           <nextGid> GId:Nat => GId +Int 1 </nextGid> 
           <grids>... (. => GId |-> cuda-grid(FCall, NBlocks, NThreads, NShared)) ...</grids>
           when GridX >Int 0 andBool GridY >Int 0
           andBool BlockX >Int 0 andBool BlockY >Int 0 andBool BlockZ >Int 0
           where NThreads = BlockX *Int BlockY *Int BlockZ
           and NBlocks = GridX *Int GridY

      rule <streamContents> cuda-next-grid(GId:Nat) =>
            cuda-launch-threads(GId, NBlocks -Int 1, NThreads -Int 1) ...</streamContents>
            <grids>... GId |-> cuda-grid(_, NBlocks:Nat, NThreads:Nat, _) ...</grids>
 
       rule [cuda-launch-kernel-thread]:
            <streamContents> cuda-launch-kernel-thread(TBody:K, GId:Nat, BId:Nat, TId:Nat) 
                  => cuda-launch-thread(
                        (cuda-alloc-shared(Identifier("$dynamic"), NShared:Nat) 
                              ~> cuda-put-blockIdx(cuda-new-dim3(cuda-dim1-to-dim3(BId:Nat, GridDim:KResult)))
                              ~> cuda-put-threadIdx(cuda-new-dim3(cuda-dim1-to-dim3(TId:Nat, BlockDim:KResult)))
                              ~> cuda-sync-grid
                              ~> TBody:K),
                        GId:Nat, BId:Nat, TId:Nat, 
                        cuda-access-device-read cuda-access-device-write) 
            ...</streamContents>
            <grids>... GId |-> cuda-grid(_, _, _, NShared:Nat) ...</grids>
            <cudaGridDim>... GId:Nat |-> cuda-dim-info(_, GridDim:KResult) ...</cudaGridDim>
            <cudaBlockDim>... GId:Nat |-> cuda-dim-info(_, BlockDim:KResult) ...</cudaBlockDim>
       rule [cuda-launch-thread]:
            <stream>...
                  <sid> SId:Nat </sid>
                  <streamContents> cuda-launch-thread(TBody:K, GId:Nat, BId:Nat, TId:Nat, Access:Set) 
                        => . 
                  ...</streamContents>
            ...</stream>
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
                  <sid> SId:Nat </sid>
                  <gid> GId:Nat </gid>
                  <bid> BId:Nat </bid>
                  <tid> TId:Nat </tid> 
                  <threadAccess> Access:Set </threadAccess>
                  <nextLoc> loc(threadId(ThreadId) +Int 0, 0, 0) </nextLoc>
                  <threadId> ThreadId </threadId>
                  C:Bag
                  <k> TBody:K </k>
                  <threadLocal>
                        <callStack> .List </callStack>
                        <control>
                              C':Bag
                        </control>
                  </threadLocal>
            </thread>)
            [computational]


      /*@ BlockIdx/ThreadIdx initialization. */

      syntax K ::= "cuda-put-blockIdx" "(" K ")" [strict]
                 | "cuda-put-threadIdx" "(" K ")" [strict]

      rule [cuda-put-blockIdx]:
            <k> cuda-put-blockIdx(Loc:KResult) => . ...</k>
            <blockIdx> _ => Loc:KResult </blockIdx>
      rule [cuda-put-threadIdx]:
            <k> cuda-put-threadIdx(Loc:KResult) => . ...</k>
            <threadIdx> _ => Loc:KResult </threadIdx>

      syntax K ::= "cuda-dim1-to-dim3" "(" Nat "," K ")" [strict(2)]
                 | "cuda-init-dim3" "(" K "," K ")" [strict]

      rule [cuda-dim1-to-dim3]:
            <k> cuda-dim1-to-dim3(TId:Nat, cuda-dim3(DX:Nat, DY:Nat, DZ:Nat)) 
                  => cuda-dim3(D0, D1, D2)
            ...</k>
            where D0 = TId /Int (DY *Int DZ)
              and D1 = (TId -Int (D0 *Int DY *Int DZ)) /Int DZ
              and D2 = TId -Int (D0 *Int DY *Int DZ) -Int (D1 *Int DZ)

      rule [cuda-init-dim3]:
            <k> cuda-init-dim3(Loc:KResult, cuda-dim3(DX:Nat, DY:Nat, DZ:Nat)) 
                  => Computation(Loc:KResult . Identifier("x") := DX:Nat)
                  ~> Computation(Loc:KResult . Identifier("y") := DY:Nat)
                  ~> Computation(Loc:KResult . Identifier("z") := DZ:Nat)
            ...</k>

      syntax K ::= "cuda-set-thread-access" "(" Set ")"
      rule [cuda-set-thread-access]:
            <k> cuda-set-thread-access(Access:Set) => . ...</k>
            <threadAccess> _:Set => Access:Set </threadAccess>

      syntax K ::= "cuda-fake-access" "(" K "," Set ")"
      rule [cuda-fake-access]:
            <k> cuda-fake-access(K:K, Access:Set)
                  => cuda-set-thread-access(Access:Set)
                  ~> K:K
                  ~> cuda-set-thread-access(OldAccess:Set)
            ... </k>
            <threadAccess> OldAccess:Set </threadAccess>

      syntax K ::= "cuda-new-dim3" "(" K ")" [strict]
      rule [cuda-new-dim3]:
            <k> cuda-new-dim3(Dim:KResult)
                 => allocateType(loc(Base:Nat, 0, 0), cuda-dim3-type)
                 ~> cuda-fake-access(
                        cuda-init-dim3(lv(loc(Base:Nat, 0, 0), unqualifyType(cuda-dim3-type)), Dim:KResult), 
                        cuda-access-device-read cuda-access-device-write)
                 ~> lv(loc(Base:Nat, 0, 0), unqualifyType(cuda-dim3-type))
            ...</k>
            <freshNat> Fresh:Nat => Fresh:Nat +Int 1 </freshNat>
            where Base = threadId(allocatedDuration) +Int Fresh
            
       /*@ Launch the first thread of a grid. This thread is responsible for
        clearing the stream when it's done executing.*/

      rule [cuda-launch-grid-head]:
            <streamContents> 
                  (. => cuda-launch-kernel-thread((Computation(FCall) ~> cuda-sync-grid ~> cuda-free-shared ~> cuda-join(GId)), GId, 0, 0))
                  ~> (cuda-launch-threads(GId:Nat, 0, 0) => cuda-join(GId))
            ...</streamContents>
            <grids> ... GId |-> cuda-grid(FCall:K, NBlocks:Nat, NThreads:Nat, NShared:Nat) ... </grids>
            [structural]

      /*@ Launch the first thread of a block. */

      rule [cuda-launch-block-head]:
            <streamContents> 
                  (. => cuda-launch-kernel-thread((Computation(FCall) ~> cuda-sync-grid), GId, BId, 0)) 
                  ~> cuda-launch-threads(GId:Nat, (BId:Nat => BId -Int 1), (0 => NThreads -Int 1)) 
            ...</streamContents>
            <grids> ... GId |-> cuda-grid(FCall:K, NBlocks:Nat, NThreads:Nat, NShared:Nat) ... </grids>
            when BId >Int 0
            [structural]

      /*@ Launch the other threads. */

      rule [cuda-launch-other-threads]:
            <streamContents> 
                  (. => cuda-launch-kernel-thread((Computation(FCall) ~> cuda-sync-grid), GId, BId, TId)) 
                  ~> cuda-launch-threads(GId:Nat, BId:Nat, (TId:Nat => TId -Int 1)) 
            ...</streamContents>
            <grids> ... GId |-> cuda-grid(FCall:K, NBlocks:Nat, NThreads:Nat, NShared:Nat) ... </grids>
            when TId >Int 0
            [structural]

      /*@ Identifiers accessible from device functions. */

      rule [threadIdx]:
            <k> Identifier("threadIdx") => Loc:KResult ...</k>
            <threadIdx> Loc:KResult </threadIdx>
      rule [blockIdx]:
            <k> Identifier("blockIdx") => Loc:KResult ...</k>
            <blockIdx> Loc:KResult </blockIdx>
      rule [gridDim]:
            <k> Identifier("gridDim") => Loc:KResult ...</k>
            <gid> GId:Nat </gid>
            <cudaGridDim> ... GId:Nat |-> cuda-dim-info(Loc:KResult, _) ... </cudaGridDim>
      rule [blockDim]:
            <k> Identifier("blockDim") => Loc:KResult ...</k>
            <gid> GId:Nat </gid>
            <cudaBlockDim> ... GId:Nat |-> cuda-dim-info(Loc:KResult, _) ... </cudaBlockDim>
      rule [warpSize]:
            <k> Identifier("warpSize") => tv(cuda-cfg-warpSize, t(.Set, int)) ...</k>

      /*@ \subsection{Block-level synchronization} */

      /*@ Start the token at thread 0. */

      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> 0 </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid> 
            <k> prepareBuiltin((Identifier("__syncthreads")), _) 
                => cuda-sync(1, GId, BId, 0, NThreads, 1) 
            ...</k>
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> TId:Nat </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid>
            <k> prepareBuiltin((Identifier("__syncthreads")), _) 
                => cuda-sync(0, GId, BId, TId, NThreads, 1) 
            ...</k>
            when TId >Int 0

      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> 0 </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid> 
            <k> prepareBuiltin((Identifier("__syncthreads_count")), tv(0, _))
                => cuda-sync(1, GId, BId, 0, NThreads, 0)
            ...</k>
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> 0 </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid> 
            <k> prepareBuiltin((Identifier("__syncthreads_count")), tv(Pred:Nat, _))
                => cuda-sync(1, GId, BId, 0, NThreads, 1)
            ...</k>
            when Pred:Nat =/=Int 0
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> TId:Nat </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid>
            <k> prepareBuiltin((Identifier("__syncthreads_count")), tv(0, _)) 
                => cuda-sync(0, GId, BId, TId, NThreads, 0)
            ...</k>
            when TId >Int 0
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> TId:Nat </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid>
            <k> prepareBuiltin((Identifier("__syncthreads_count")), tv(Pred:Nat, _)) 
                => cuda-sync(0, GId, BId, TId, NThreads, 1)
            ...</k>
            when TId >Int 0
            andBool Pred:Nat =/=Int 0

      syntax K ::= "cuda-sync-type-and"

      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> 0 </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid> 
            <k> prepareBuiltin((Identifier("__syncthreads_and")), tv(0, _))
                => cuda-sync(1, GId, BId, 0, NThreads, 0) ~> cuda-sync-type-and
            ...</k>
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> 0 </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid> 
            <k> prepareBuiltin((Identifier("__syncthreads_and")), tv(Pred:Nat, _))
                => cuda-sync(1, GId, BId, 0, NThreads, 1) ~> cuda-sync-type-and
            ...</k>
            when Pred:Nat =/=Int 0
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> TId:Nat </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid>
            <k> prepareBuiltin((Identifier("__syncthreads_and")), tv(0, _)) 
                => cuda-sync(0, GId, BId, TId, NThreads, 0) ~> cuda-sync-type-and 
            ...</k>
            when TId >Int 0
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> TId:Nat </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid>
            <k> prepareBuiltin((Identifier("__syncthreads_and")), tv(Pred:Nat, _)) 
                => cuda-sync(0, GId, BId, TId, NThreads, 1) ~> cuda-sync-type-and 
            ...</k>
            when TId >Int 0
            andBool Pred:Nat =/=Int 0

      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <k> (tv(Count:Nat, _) ~> cuda-sync-type-and) 
                  => tv(1, t(.Set, int)) 
            ...</k>
            when Count:Nat ==Int NThreads:Nat
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <k> (tv(Count:Nat, _) ~> cuda-sync-type-and) 
                  => tv(0, t(.Set, int)) ...</k>
            when Count:Nat =/=Int NThreads:Nat

      /*@ __syncthreads_or() and __syncthreads_count() are the
        same. */
      // TODO: Maybe _or should just be a macro?
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> 0 </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid> 
            <k> prepareBuiltin((Identifier("__syncthreads_or")), tv(0, _))
                => cuda-sync(1, GId, BId, 0, NThreads, 0)
            ...</k>
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> 0 </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid> 
            <k> prepareBuiltin((Identifier("__syncthreads_or")), tv(Pred:Nat, _))
                => cuda-sync(1, GId, BId, 0, NThreads, 1)
            ...</k>
            when Pred:Nat =/=Int 0
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> TId:Nat </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid>
            <k> prepareBuiltin((Identifier("__syncthreads_or")), tv(0, _)) 
                => cuda-sync(0, GId, BId, TId, NThreads, 0)
            ...</k>
            when TId >Int 0
      rule <grids>... GId |-> cuda-grid(_, _, NThreads:Nat, _) ...</grids>
            <tid> TId:Nat </tid> <gid> GId:Nat </gid> <bid> BId:Nat </bid>
            <k> prepareBuiltin((Identifier("__syncthreads_or")), tv(Pred:Nat, _)) 
                => cuda-sync(0, GId, BId, TId, NThreads, 1)
            ...</k>
            when TId >Int 0
            andBool Pred:Nat =/=Int 0

      /*@ Pass the token up. */

      rule [cuda-syncthreads-passup]:
            <k> cuda-sync(1 => 0, GId:Nat, BId:Nat, TId:Nat, NThreads:Nat, PredTotal:Nat) ...</k>
            <k> cuda-sync(0 => 1, GId, BId, SuccTId:Nat, NThreads, Pred:Nat => Pred:Nat +Int PredTotal:Nat) ...</k>
            when SuccTId ==Int TId +Int 1

      /*@ Pivot. */

      rule [cuda-syncthreads-pivot]:
            <k> (. => cuda-rc-sync-shared) ~> cuda-sync(1 => 2, _, _, TId:Nat, NThreads:Nat, _) ...</k> 
            when NThreads ==Int TId +Int 1

      /*@ Pass it back down. */

      rule [cuda-syncthreads-passdown]:
            <k> cuda-sync(2, GId:Nat, BId:Nat, SuccTId:Nat, NThreads:Nat, PredTotal:Nat) => tv(PredTotal:Nat, t(.Set, int)) ...</k>
            <k> cuda-sync(0 => 2, GId, BId, TId:Nat, NThreads, _ => PredTotal:Nat) ...</k>
            when SuccTId ==Int TId +Int 1
      rule <k> cuda-sync(2, _, _, 0, _, PredTotal:Nat) => tv(PredTotal:Nat, t(.Set, int)) ...</k> 

      /*@ \subsection{Grid-level synchronization} 
      This isn't an operation supported by CUDA, but we need it in this model to make
      the streams synchronous. Same deal as syncing threads within a block. This
      shouldn't ever clash with a block-level sync because if there are multiple
      blocks in the grid (the only time a clash would be possible), then NThreads
      will be different for the grid-level and block-level sync.*/

      rule [cuda-sync-grid-head]:
            <k> cuda-sync-grid => Computation(cuda-sync(1, GId, 0, 0, NThreads *Int NBlocks, 1)) ...</k>
            <gid> GId:Nat </gid> <bid> 0 </bid> <tid> 0 </tid>
            <grids>... GId |-> cuda-grid(_, NBlocks:Nat, NThreads:Nat, _) ...</grids>
      rule [cuda-sync-grid]:
            <k> cuda-sync-grid => Computation(cuda-sync(0, GId, 0, (NThreads *Int BId) +Int TId, NThreads *Int NBlocks, 1)) ...</k>
            <gid> GId:Nat </gid> <bid> BId:Nat </bid> <tid> TId:Nat </tid>
            <grids>... GId |-> cuda-grid(_, NBlocks:Nat, NThreads:Nat, _) ...</grids>
            when BId >Int 0 orBool TId >Int 0

      // Cleanup completed threads. 
      // TODO: probably other stuff needs to be done here (free memory,
      // perhaps?)
      rule (<thread>... <k>.K</k> ...</thread> => .)

end module
