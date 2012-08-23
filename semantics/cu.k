load cu-syntax
load dynamic-c-semantics

load cuda-semantics

module CU-SEMANTICS is
      including CUDA-SEMANTICS
end module

module C-SEMANTICS is
	including DYNAMIC-C-CONFIGURATION
	including DYNAMIC-C-SEMANTICS
end module

module CU is 
	including C-SEMANTICS 
      including CU-SEMANTICS
end module
