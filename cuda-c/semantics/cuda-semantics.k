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

	rule [ignore-cuda-global]: t(S:Set, qualifiedType(t(S':Set, T:K), CudaGlobal)) 
		=> t(S:Set S':Set, T:K) 
		[structural anywhere]
	rule [ignore-cuda-device]: t(S:Set, qualifiedType(t(S':Set, T:K), CudaDevice)) 
		=> t(S:Set S':Set, T:K) 
		[structural anywhere]
	rule [ignore-cuda-host]: t(S:Set, qualifiedType(t(S':Set, T:K), CudaHost)) 
		=> t(S:Set S':Set, T:K) 
		[structural anywhere]

      // Actually seems like this should just be void.
      rule <type> CudaSpawn(T:KResult, _) => innerType(T) ...</type>
end module

module CUDA-SEMANTICS is
      including DYNAMIC-C-SEMANTICS
      including CUDA-TYPING


      //rule <k> CudaSpawn(C:K, _) => C ... </k>
end module


