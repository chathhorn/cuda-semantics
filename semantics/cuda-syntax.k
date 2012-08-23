load common-c-syntax

module CUDA-SYNTAX is
      including COMMON-C-SYNTAX

	syntax C ::= CudaSpecifier
	
	syntax CudaSpecifier ::= "CudaDevice"
	syntax CudaSpecifier ::= "CudaGlobal"
	syntax CudaSpecifier ::= "CudaHost"
	syntax CudaSpecifier ::= "CudaNoinline"
	syntax CudaSpecifier ::= "CudaConstant"
	syntax CudaSpecifier ::= "CudaShared"

      syntax SpecifierElem ::= CudaSpecifier

      // call(), nblocks, nthreads
      syntax Expression ::= "CudaSpawn2" "(" K "," K "," K ")" [metadata "type-strict=(1)" klabel(CudaSpawn2) strict(2 3)]
	context CudaSpawn2(Call((HOLE => reval(HOLE)), _), _, _) [ndheat]
	context CudaSpawn2(Call(_, List((_:List{K},, (HOLE => reval(HOLE)),, _:List{K}))), _, _) [ndheat]
	context CudaSpawn2(_, (HOLE => reval(HOLE)), _) [ndheat]
	context CudaSpawn2(_, _, (HOLE => reval(HOLE))) [ndheat]

      // call(), nblocks, nthreads, shared
      syntax Expression ::= "CudaSpawn3" "(" K "," K "," K "," K ")" [metadata "type-strict=(1)" klabel(CudaSpawn3) strict(2 3 4)]
	context CudaSpawn3(Call((HOLE => reval(HOLE)), _), _, _, _) [ndheat]
	context CudaSpawn3(Call(_, List((_:List{K},, (HOLE => reval(HOLE)),, _:List{K}))), _, _, _) [ndheat]
	context CudaSpawn3(_, (HOLE => reval(HOLE)), _, _) [ndheat]
	context CudaSpawn3(_, _, (HOLE => reval(HOLE)), _) [ndheat]
	context CudaSpawn3(_, _, _, (HOLE => reval(HOLE))) [ndheat]

      // call(), nblocks, nthreads, shared, stream
      syntax Expression ::= "CudaSpawn4" "(" K "," K "," K "," K "," K ")" [metadata "type-strict=(1)" klabel(CudaSpawn4) strict(2 3 4 5)]
	context CudaSpawn4(Call((HOLE => reval(HOLE)), _), _, _, _, _) [ndheat]
	context CudaSpawn4(Call(_, List((_:List{K},, (HOLE => reval(HOLE)),, _:List{K}))), _, _, _, _) [ndheat]
	context CudaSpawn4(_, (HOLE => reval(HOLE)), _, _, _) [ndheat]
	context CudaSpawn4(_, _, (HOLE => reval(HOLE)), _, _) [ndheat]
	context CudaSpawn4(_, _, _, (HOLE => reval(HOLE)), _) [ndheat]
	context CudaSpawn4(_, _, _, _, (HOLE => reval(HOLE))) [ndheat]

end module
