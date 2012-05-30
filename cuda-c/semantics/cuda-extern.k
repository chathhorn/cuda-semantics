load common-c-semantics
module CUDA-EXTERN is
      // TODO figure out the dependencies.
      including COMMON-SEMANTIC-SYNTAX
      declare isCudaSharedType : Type -> Bool

      syntax K ::= "cuda-decl-function" "(" Id "," K ")"
      syntax K ::= "cuda-check-isdevice" "(" Nat ")" 
      syntax K ::= "cuda-check-ishost" "(" Nat ")" 
end module
