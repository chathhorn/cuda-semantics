load c-syntax
load dynamic-c-semantics

load cuda-semantics

module C-SEMANTICS is
	including DYNAMIC-C-CONFIGURATION
	including DYNAMIC-C-SEMANTICS
      including CUDA-SEMANTICS
end module

module C is 
	including C-SEMANTICS 
end module
