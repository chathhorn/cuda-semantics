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

//   rule <k> culaunchkernel(SId:Nat, NBlocks:Nat, NThreads:Nat, NShared:Nat, Xs:Ids, Vs:Vals, S:K) => true ...</k>
//        <nextGid> GId:Nat => GId +Int 1 </nextGid> 
//        <grids>... (. => GId |-> cugrid(NBlocks, NThreads, NShared, Xs, Vs, S)) ...</grids>
//        <stream>... <sid> SId </sid> 
//              <streamContents> ... (. => cunextgrid(GId)) </streamContents> 
//              ...</stream>
//   rule <k> culaunchkernel(SId:Nat, NBlocks:Nat, NThreads:Nat, NShared:Nat, Xs:Ids, Vs:Vals, S:K) => true ...</k>
//        <nextGid> GId:Nat => GId +Int 1 </nextGid> 
//        <grids>... (. => GId |-> cugrid(NBlocks, NThreads, NShared, Xs, Vs, S)) ...</grids>
//        <streams> Streams:Set (. => SetItem(SId)) </streams>
//        (. => <stream>... <sid> SId </sid> 
//              <streamContents> cunextgrid(GId) </streamContents> 
//              ...</stream>)
//        when notBool(SId in Streams)
// 
//   rule <streamContents> cunextgrid(GId:Nat) =>
//        culaunchthreads(GId, NBlocks -Int 1, NThreads -Int 1) ...</streamContents>
//        <grids>... GId |-> cugrid(NBlocks:Nat, NThreads:Nat, _, _, _, _) ...</grids>
// 
// /*@ Launch the first thread of a grid. This thread is responsible for clearing
// the stream when it's done executing.*/
// 
//   syntax K ::= "sharedLoc" "(" Nat "," Nat ")"
// 
//   rule <streamContents> culaunchthreads(GId:Nat, 0, 0) => cujoin(GId) ...</streamContents>
//        <grids> ... GId |-> cugrid(NBlocks:Nat, NThreads:Nat, NShared:Nat, Xs:Ids, Vs:Vals, S:K) ... </grids>
//        <genv> Env:Map </genv>
//        <nextLoc> L:Nat => L +Int 1 +Int NShared </nextLoc>
//        <shared>... . => sharedLoc(GId, 0) |-> L ... </shared>
//        <devStore>... . => L |-> arrayRef(L +Int 1, NShared)
//                        L +Int 1 .. L +Int NShared |-> 0 ...</devStore> // zeroing shared memory.
//       (. => <thread>... <k> cufillenv ~> S ~> cusyncgrid ~> cujoin(GId) </k> 
//             <gid> GId </gid>
//             <bid> 0 </bid>
//             <tid> 0 </tid> 
//             <env> Env </env> ...</thread>)
// 
// /*@ Launch the first thread of a block. */
// 
//   rule <streamContents> culaunchthreads(GId:Nat, (BId:Nat => BId -Int 1), (0 => NThreads -Int 1)) ...</streamContents>
//        <grids> ... GId |-> cugrid(NBlocks:Nat, NThreads:Nat, NShared:Nat, Xs:Ids, Vs:Vals, S:K) ... </grids>
//        <genv> Env:Map </genv>
//        <nextLoc> L:Nat => L +Int 1 +Int NShared </nextLoc>
//        <shared>... . => sharedLoc(GId, BId) |-> L ... </shared>
//        <devStore>... . => L |-> arrayRef(L +Int 1, NShared)
//                        L +Int 1 .. L +Int NShared |-> 0 ...</devStore> // zeroing shared memory.
//       (. => <thread>... <k> cufillenv ~> S ~> cusyncgrid </k> 
//             <gid> GId </gid>
//             <bid> BId </bid>
//             <tid> 0 </tid> 
//             <env> Env </env> ...</thread>)
//        when BId >Int 0
// 
// /*@ Launch the other threads. */
// 
//   rule <streamContents> culaunchthreads(GId:Nat, BId:Nat, (TId:Nat => TId -Int 1)) ...</streamContents>
//        <grids> ... GId |-> cugrid(NBlocks:Nat, NThreads:Nat, NShared:Nat, Xs:Ids, Vs:Vals, S:K) ... </grids>
//        <genv> Env:Map </genv>
//       (. => <thread>... <k> cufillenv ~> S ~> cusyncgrid </k> 
//             <gid> GId </gid>
//             <bid> BId </bid>
//             <tid> TId </tid> 
//             <env> Env </env> ...</thread>)
//        when TId >Int 0
// 
//   syntax K ::= "cufillenv"
//   rule <gid> GId:Nat </gid>
//        <bid> BId:Nat </bid>
//        <tid> TId:Nat </tid> 
//        <grids> ... GId |-> cugrid(NBlocks:Nat, NThreads:Nat, _, Xs:Ids, Vs:Vals, S:K) ... </grids>
//        <shared> ... sharedLoc(GId, BId) |-> L:Nat ... </shared>
//        <k> cufillenv => bindto(("blockIdx", "threadIdx", "gridDim", "blockDim", Xs), (BId, TId, NBlocks, NThreads, Vs)) ... </k>
//       <env> Env => Env[L/"shared"] </env>
//     //  rule [cuda-kernel-call]:
//     //        <k> CudaSpawn2(Call(X:Id, Args:K), NBlocks:K, NThreads:K) 
//     //              => reval(X) ~> CudaSpawn2(Call(HOLE, Args), NBlocks, NThreads)
//     //        ...</k>
//     //        [structural]

// 
// 

  // call, nblocks, threads per block, nshared, stream
  syntax K ::= "culaunchkernel" "(" K "," Nat "," Nat "," Nat "," Nat ")"
             // grid id, blocks left to spawn, threads left to spawn.
             | "culaunchthreads" "(" Nat "," Nat "," Nat ")"
             // grid id
             | "cunextgrid" "(" Nat ")"
             // call, nblocks, nthreads, nshared
             | "cugrid" "(" K "," Nat "," Nat "," Nat ")"
             // Location of shared memory in the store.
             | "cublock" "(" Nat ")"
             | "cujoin" "(" Nat ")"

       // rule [cuda-spawn2]:
       //       <k> CudaSpawn2(Call(tv(Loc:Nat, t(_, pointerType(T:KResult))), List(L:List{KResult})), _, _)
       //             => application(readFunction(Loc:Nat), L:List{KResult})
       //       ...</k>
       //       when isFunctionType(T:KResult)
       //       [structural]

       rule [cuda-spawn2]:
             <k> CudaSpawn2(Fun:K, tv(NBlocks:Nat, _), tv(NThreads:Nat, _))
                   => culaunchkernel(Fun, NBlocks, NThreads, 0, 0)
             ...</k>
             <gid> 0 </gid>
             [structural]

       rule [cuda-spawn3]:
             <k> CudaSpawn3(Fun:K, tv(NBlocks:Nat, _), tv(NThreads:Nat, _), tv(NShared:Nat, _))
                   => culaunchkernel(Fun, NBlocks, NThreads, NShared, 0)
             ...</k>
             <gid> 0 </gid>
             [structural]

       rule [cuda-spawn4]:
             <k> CudaSpawn4(Fun:K, tv(NBlocks:Nat, _), tv(NThreads:Nat, _), tv(NShared:Nat, _), tv(SId:Nat, _))
                   => culaunchkernel(Fun, NBlocks, NThreads, NShared, SId)
             ...</k>
             <gid> 0 </gid>
             [structural]

       rule < k => finalComputation > culaunchkernel(_, 0, _, _, _) ...</ k => finalComputation > 
		(.Bag => <errorCell> Error("90000", "NBlocks zero.") </errorCell>)
       rule < k => finalComputation > culaunchkernel(_, _, 0, _, _) ...</ k => finalComputation > 
		(.Bag => <errorCell> Error("90001", "Nthreads zero.") </errorCell>)
end module


