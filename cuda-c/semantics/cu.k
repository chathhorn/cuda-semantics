load c-syntax
load dynamic-c-semantics

load cuda-semantics
load cuda-typing

module CU-SEMANTICS is
      including CUDA-SEMANTICS
      including CUDA-TYPING
end module

module C-SEMANTICS is
	including DYNAMIC-C-CONFIGURATION
	including DYNAMIC-C-SEMANTICS
end module

module CU is 
	including C-SEMANTICS 
      including CU-SEMANTICS
end module
