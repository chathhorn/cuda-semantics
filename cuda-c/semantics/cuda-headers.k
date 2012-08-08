load common-c-typing

module CUDA-DECLARATIONS-HEADER is
      syntax K ::= "cuda-check-isglobal" "(" K ")" 
      syntax K ::= "cuda-check-isdevice" "(" Nat ")" 
      syntax K ::= "cuda-check-ishost" "(" Nat ")" 
      syntax K ::= "cuda-decl-function" "(" Id "," K ")"
end module

module CUDA-TYPING-HEADER is
      including COMMON-C-TYPING

      declare isCudaSharedType : Type -> Bool
      declare isCudaHostType : Type -> Bool
      declare isCudaDeviceType : Type -> Bool
      declare isCudaGlobalType : Type -> Bool
end module

module CUDA-MEMORY-HEADER is
      syntax K ::= "cuda-set-props" "(" Nat "," K ")"
      syntax K ::= "cuda-set-mdevice" "(" Nat ")"
      // gid, bid
      syntax K ::= "cuda-shared-loc" "(" Nat "," Nat "," Id ")"
      syntax K ::= "cuda-alloc-shared" "(" Id "," Nat ")"
      syntax K ::= "cuda-free-shared"

      syntax BagItem ::= "cuda-mdevice"
                       | "cuda-mshared"
                       | "cuda-mhost"

      syntax SetItem ::= "cuda-access-device-read"
                       | "cuda-access-device-write"
                       | "cuda-access-host-read"
                       | "cuda-access-host-write"

	syntax K ::= "cuda-read-check" "(" K ")"
	syntax K ::= "cuda-write-check" "(" K ")"

      syntax K ::= "cuda-std-malloc-check" "(" Nat ")"
end module

module CUDA-STREAMS-HEADER is
      syntax K ::= "cuda-put-in-stream" "(" K "," Nat ")" 
                 | "cuda-join" "(" Nat ")"
end module

module CUDA-THREADS-HEADER is
      syntax K ::= "cuda-launch-thread" "(" K "," Nat "," Nat "," Nat "," Set ")"
                 | "cuda-grid" "(" K "," Nat "," Nat "," Nat ")"
end module

module CUDA-DEVICES-HEADER is
      syntax K ::= "cuda-device-synchronize"
end module

module CUDA-RACECHECK-HEADER is
      syntax K ::= "cuda-rc-read" "(" Nat ")"
                 | "cuda-rc-write" "(" Nat ")"
      syntax K ::= "cuda-rc-sync-shared"
                 | "cuda-rc-sync-global"
end module
