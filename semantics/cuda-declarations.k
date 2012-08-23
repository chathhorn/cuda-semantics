load dynamic-c-semantics
load cuda-headers

module CUDA-DECLARATIONS is
      including CUDA-DECLARATIONS-HEADER
      including DYNAMIC-C-SEMANTICS
      including CUDA-TYPING-HEADER
      including CUDA-MEMORY-HEADER

      declare incompleteToPointerType : Type -> Type
	define incompleteToPointerType(t(S:Set, incompleteArrayType(T:KResult))) => t(S:Set, pointerType(T:KResult))
	define incompleteToPointerType(t(S:Set, qualifiedType(T:KResult, K:K))) => t(S:Set, qualifiedType(incompleteToPointerType(T:KResult), K:K))
	define incompleteToPointerType(t(S:Set, T:K)) => t(S:Set, T:K)
		when getKLabel(T:K) =/=KLabel 'incompleteArrayType
		andBool getKLabel(T:K) =/=KLabel 'qualifiedType 

      /*@ Declarations. */
      // E.g., extern __shared__ int shared[];
      // C.f., extern __shared__ int* shared; (not legal)
      // TODO: This is pretty hackish, I just convert incomplete array types to
      // pointer types.
       rule [cuda-declare-shared-dynamic]:
		<k> declareInternalVariable(X:Id, T:KResult, NoInit)
			=> declareWithLinkage(X:Id, incompleteToPointerType(T:KResult), SingleInit(tv(SharedLoc:Nat, unqualifyType(incompleteToPointerType(T:KResult)))), noLinkage)
		...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaShared>... 
                  cuda-shared-loc(GId:Nat, BId:Nat, Identifier("$dynamic")) |-> SharedLoc:Nat 
            ...</cudaShared>
            <cudaSharedObjects> ... (. => ListItem(SharedLoc)) </cudaSharedObjects>
		when notBool isStaticType(T:KResult)
		andBool isExternType(T:KResult)
            andBool isCudaSharedType(T:KResult)
            // __shared__ and extern on anything but a pointer must be illegal.
            andBool isIncompleteType(T:KResult)
		[structural] 

       // TODO: Initializers (is that legal on shared variables?).
       // E.g., __shared__ int shared[10]; but some other thread has already
       // allocated the shared memory.
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
            <cudaSharedObjects> ... (. => ListItem(SharedLoc)) </cudaSharedObjects>
		when notBool isStaticType(T:KResult)
		andBool notBool isExternType(T:KResult)
            andBool isCudaSharedType(T:KResult)
		[structural] 
       // E.g., __shared__ int shared[10]; 
       rule [cuda-declare-shared]:
		<k> declareInternalVariable(X:Id, T:KResult, NoInit)
			=> allocateType(loc(Base:Nat, 0, 0), T:KResult)
                  ~> addToEnv(X:Id, loc(Base:Nat, 0, 0))
			~> giveType(X:Id, T:KResult)
		...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaShared>... 
                  . => (cuda-shared-loc(GId:Nat, BId:Nat, X:Id) |-> loc(Base:Nat, 0, 0))
            ...</cudaShared>
            <cudaSharedObjects> ... (. => ListItem(loc(Base:Nat, 0, 0))) </cudaSharedObjects>
            <freshNat> Fresh:Nat => Fresh:Nat +Int 1 </freshNat>
		when notBool isStaticType(T:KResult)
		andBool notBool isExternType(T:KResult)
            andBool isCudaSharedType(T:KResult)
            where Base = threadId(allocatedDuration) +Int Fresh
		[structural] 

      rule [cuda-decl-function]:
            <k> cuda-decl-function(X:Id, T:KResult) 
                => cuda-decl-function-host(X:Id, T:KResult) 
                ~> cuda-decl-function-device(X:Id, T:KResult) 
                ~> cuda-decl-function-global(X:Id, T:KResult) 
            ...</k>

      syntax K ::= "cuda-decl-function-host" "(" Id "," KResult ")"
      rule [cuda-decl-function-host]:
            <k> cuda-decl-function-host(X:Id, T:KResult) => . ...</k>
            <cudaHostFunctions> Fs:Set (. => SetItem(X:Id)) </cudaHostFunctions>
            when isCudaHostType(T:KResult)
            andBool notBool (X:Id in Fs:Set)
      rule [cuda-decl-function-host-not]:
            <k> cuda-decl-function-host(X:Id, T:KResult) => . ...</k>
            <cudaHostFunctions> Fs:Set </cudaHostFunctions>
            when notBool isCudaHostType(T:KResult)
            orBool X:Id in Fs:Set

      syntax K ::= "cuda-decl-function-device" "(" Id "," KResult ")"
      rule [cuda-decl-function-device]:
            <k> cuda-decl-function-device(X:Id, T:KResult) => . ...</k>
            <cudaDeviceFunctions> Fs:Set (. => SetItem(X:Id)) </cudaDeviceFunctions>
            when isCudaDeviceType(T:KResult)
            andBool notBool (X:Id in Fs:Set)
      rule [cuda-decl-function-device-not]:
            <k> cuda-decl-function-device(X:Id, T:KResult) => . ...</k>
            <cudaDeviceFunctions> Fs:Set </cudaDeviceFunctions>
            when notBool isCudaDeviceType(T:KResult)
            orBool X:Id in Fs:Set

      syntax K ::= "cuda-decl-function-global" "(" Id "," KResult ")"
      rule [cuda-decl-function-global]:
            <k> cuda-decl-function-global(X:Id, T:KResult) => . ...</k>
            <cudaGlobalFunctions> Fs:Set (. => SetItem(X:Id)) </cudaGlobalFunctions>
            when isCudaGlobalType(T:KResult)
            andBool notBool (X:Id in Fs:Set)
      rule [cuda-decl-function-global-not]:
            <k> cuda-decl-function-global(X:Id, T:KResult) => . ...</k>
            <cudaGlobalFunctions> Fs:Set </cudaGlobalFunctions>
            when notBool isCudaGlobalType(T:KResult)
            orBool X:Id in Fs:Set

      // Note that these take a pointer and not the whole Call().
      rule [cuda-check-isglobal-pass]:
           <k> cuda-check-isglobal(Call(tv(Loc:Nat, _), _)) => . ...</k>
           <functions>... Loc:Nat |-> functionObject(X:Id, _, _) ...</functions>
           <cudaGlobalFunctions>... SetItem(X:Id) ...</cudaGlobalFunctions>
           [structural]
      rule [cuda-check-isglobal-fail]:
           < k => finalComputation > cuda-check-isglobal(Call(tv(Loc:Nat, _), _)) ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90000", "CUDA: Attempting to call a device function not marked __global__ from the host: " +String S:String) </errorCell>)
           <functions>... Loc:Nat |-> functionObject(Identifier(S:String), _, _) ...</functions>
           <cudaGlobalFunctions> Fs:Set </cudaGlobalFunctions>
           when notBool(Identifier(S:String) in Fs:Set)
           [structural]

      /*@ A function on the device must be tagged with either __device__ or
          __global__. */

      rule [cuda-check-isdevice-pass]:
           <k> cuda-check-isdevice(Loc:Nat) => . ...</k>
           <functions>... Loc:Nat |-> _:KLabel(X:Id,, _:List{K}) ...</functions>
           <cudaGlobalFunctions> GFs:Set </cudaGlobalFunctions>
           <cudaDeviceFunctions> DFs:Set </cudaDeviceFunctions>
           when X:Id in GFs:Set
           orBool X:Id in DFs:Set
           [structural]
      rule [cuda-check-isdevice-fail]:
           < k => finalComputation > cuda-check-isdevice(Loc:Nat) ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90001", "CUDA: Attempting to call a host function from the device: " +String S:String) </errorCell>)
           <functions>... Loc:Nat |-> _:KLabel(Identifier(S:String),, _:List{K}) ...</functions>
           <cudaGlobalFunctions> GFs:Set </cudaGlobalFunctions>
           <cudaDeviceFunctions> DFs:Set </cudaDeviceFunctions>
           when notBool(Identifier(S:String) in GFs:Set)
           andBool notBool(Identifier(S:String) in DFs:Set)
           [structural]

      /*@ A function on the host might not have a __host__ tag. Also, note that
          a function with both __host__ and __device__ is valid. */

      rule [cuda-check-ishost-pass]:
           <k> cuda-check-ishost(Loc:Nat) => . ...</k>
           <functions>... Loc:Nat |-> _:KLabel(X:Id,, _:List{K}) ...</functions>
           <cudaGlobalFunctions> GFs:Set </cudaGlobalFunctions>
           <cudaDeviceFunctions> DFs:Set </cudaDeviceFunctions>
           <cudaHostFunctions> HFs:Set </cudaHostFunctions>
           when (X:Id in HFs:Set)
           orBool (notBool(X:Id in GFs:Set) andBool notBool(X:Id in DFs:Set))
           [structural]
      rule [cuda-check-ishost-fail]:
           < k => finalComputation > cuda-check-ishost(Loc:Nat) ...</ k => finalComputation >
           (.Bag => <errorCell> Error("90002", "CUDA: Attempting to call a device function from the host: " +String S:String) </errorCell>)
           <functions>... Loc:Nat |-> _:KLabel(Identifier(S:String),, _:List{K}) ...</functions>
           <cudaGlobalFunctions> GFs:Set </cudaGlobalFunctions>
           <cudaDeviceFunctions> DFs:Set </cudaDeviceFunctions>
           <cudaHostFunctions> HFs:Set </cudaHostFunctions>
           when Identifier(S:String) in GFs:Set
           orBool Identifier(S:String) in DFs:Set
           andBool notBool (Identifier(S:String) in HFs:Set)
           [structural]

end module
