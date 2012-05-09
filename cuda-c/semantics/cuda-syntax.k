load common-c-syntax

module CUDA-SYNTAX is
      including COMMON-C-SYNTAX

	syntax C ::= CudaSpecifier
	
/* (from cabs.ml)
and cudaspec =
      CUDA_DEVICE | CUDA_GLOBAL | CUDA_HOST
*/

	syntax CudaSpecifier ::= "CudaDevice"
	syntax CudaSpecifier ::= "CudaGlobal"
	syntax CudaSpecifier ::= "CudaHost"

      syntax SpecifierElem ::= CudaSpecifier

      // probably should be strict?
      syntax Expression ::= "CudaSpawn" "(" K "," K ")" [metadata "type-strict=(1)" klabel(CudaSpawn)]
end module
