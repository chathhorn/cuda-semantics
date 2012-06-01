load dynamic-c-semantics
load cuda-extern

module CUDA-TYPING
      including COMMON-C-TYPING
      including CUDA-EXTERN

       rule <k> canonicalizeType-aux(B:Bag (BagItem(T:K) => .Bag), _, _, _, (_ (.Bag => BagItem(T:K)))) ...</k>
             when T:K ==K CudaGlobal
             orBool T:K ==K CudaDevice
             orBool T:K ==K CudaHost
             orBool T:K ==K CudaNoinline
             orBool T:K ==K CudaConstant
             orBool T:K ==K CudaShared
             [structural]
       rule <k> canonicalizeType-aux(.Bag, (T:KResult => t(.Set, qualifiedType(T:KResult, Q:K))), .Bag, .Bag, (_ (BagItem(Q:K) => .Bag))) ...</k>
             when Q:K ==K CudaGlobal
             orBool Q:K ==K CudaDevice
             orBool Q:K ==K CudaHost
             orBool Q:K ==K CudaNoinline
             orBool Q:K ==K CudaConstant
             orBool Q:K ==K CudaShared
             [structural]

   //   rule [ignore-cuda-global]: t(S:Set, qualifiedType(t(S':Set, T:K), CudaGlobal)) 
   //         => t(S:Set S':Set, T:K) 
   //         [structural anywhere]
   //   rule [ignore-cuda-device]: t(S:Set, qualifiedType(t(S':Set, T:K), CudaDevice)) 
   //         => t(S:Set S':Set, T:K) 
   //         [structural anywhere]
   //   rule [ignore-cuda-host]: t(S:Set, qualifiedType(t(S':Set, T:K), CudaHost)) 
   //         => t(S:Set S':Set, T:K) 
   //         [structural anywhere]

      declare isCudaHostType : Type -> Bool
      declare isCudaDeviceType : Type -> Bool
      declare isCudaGlobalType : Type -> Bool

      // TODO: functionType

      define [isCudaSharedType-true]: isCudaSharedType(t(_, qualifiedType(_, CudaShared))) => true 
      define [isCudaSharedType-prototype]: isCudaSharedType(t(_, prototype(T:KResult))) => isCudaSharedType(T:KResult) 
      define [isCudaSharedType-qualified]: isCudaSharedType(t(_, qualifiedType(T:KResult, Q:K))) => isCudaSharedType(T:KResult) 
            when getKLabel(Q:K) =/=KLabel 'CudaShared
      define [isCudaSharedType-false]: isCudaSharedType(t(_, T:K)) => false
            //when getKLabel(T:K) =/=KLabel 'functionType
            when getKLabel(T:K) =/=KLabel 'qualifiedType
            andBool getKLabel(T:K) =/=KLabel 'prototype 

      define [isCudaDeviceType-true]: isCudaDeviceType(t(_, qualifiedType(_, CudaDevice))) => true 
      define [isCudaDeviceType-prototype]: isCudaDeviceType(t(_, prototype(T:KResult))) => isCudaDeviceType(T:KResult) 
      define [isCudaDeviceType-qualified]: isCudaDeviceType(t(_, qualifiedType(T:KResult, Q:K))) => isCudaDeviceType(T:KResult) 
            when getKLabel(Q:K) =/=KLabel 'CudaDevice
      define [isCudaDeviceType-false]: isCudaDeviceType(t(_, T:K)) => false
            //when getKLabel(T:K) =/=KLabel 'functionType
            when getKLabel(T:K) =/=KLabel 'qualifiedType
            andBool getKLabel(T:K) =/=KLabel 'prototype 

      define [isCudaGlobalType-true]: isCudaGlobalType(t(_, qualifiedType(_, CudaGlobal))) => true 
      define [isCudaGlobalType-prototype]: isCudaGlobalType(t(_, prototype(T:KResult))) => isCudaGlobalType(T:KResult) 
      define [isCudaGlobalType-qualified]: isCudaGlobalType(t(_, qualifiedType(T:KResult, Q:K))) => isCudaGlobalType(T:KResult) 
            when getKLabel(Q:K) =/=KLabel 'CudaGlobal
      define [isCudaGlobalType-false]: isCudaGlobalType(t(_, T:K)) => false
            //when getKLabel(T:K) =/=KLabel 'functionType
            when getKLabel(T:K) =/=KLabel 'qualifiedType
            andBool getKLabel(T:K) =/=KLabel 'prototype 

      define [isCudaHostType-true]: isCudaHostType(t(_, qualifiedType(_, CudaHost))) => true 
      define [isCudaHostType-prototype]: isCudaHostType(t(_, prototype(T:KResult))) => isCudaHostType(T:KResult) 
      define [isCudaHostType-qualified]: isCudaHostType(t(_, qualifiedType(T:KResult, Q:K))) => isCudaHostType(T:KResult) 
            when getKLabel(Q:K) =/=KLabel 'CudaHost
      define [isCudaHostType-false]: isCudaHostType(t(_, T:K)) => false
    //        when getKLabel(T:K) =/=KLabel 'functionType
            andBool getKLabel(T:K) =/=KLabel 'qualifiedType
            andBool getKLabel(T:K) =/=KLabel 'prototype 

      // Actually seems like this should just be void.
      rule <type> CudaSpawn2(T:KResult, _, _) => innerType(T) ...</type>
      rule <type> CudaSpawn3(T:KResult, _, _, _) => innerType(T) ...</type>
      rule <type> CudaSpawn4(T:KResult, _, _, _, _) => innerType(T) ...</type>
end module

