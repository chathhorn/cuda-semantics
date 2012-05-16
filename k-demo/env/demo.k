module DEMO-SYNTAX
      syntax Ids ::= List{Id, ","} [prec(70) strict]
      syntax Exps ::= List{Exp, ","}

      syntax Exp ::= Int | Bool | String | Id
                   | Exp "+" Exp      [strict prec(33) gather(E e)]
                   | Exp "-" Exp      [strict prec(33) gather(E e)]
                   | Exp "*" Exp      [strict prec(31) gather(E e)]
                   | Exp "==" Exp     [strict prec(37)]
                   | Exp "!=" Exp     [strict prec(37)]
                   | Exp "and" Exp    [strict prec(55) gather(E e)]
                   | Exp "or" Exp     [strict prec(59) gather(E e)]
                   | "not" Exp        [strict prec(53)]
                   | Exp "(" Exps ")" [strict prec(2)]
                   | "read" "(" ")"
                        
      syntax Stmt ::= "{" "}"
                    | "{" Stmts "}"
                    | Exp ";"                          [strict prec(45)]
                    | "if" Exp "then" Stmt "else" Stmt [strict(1) prec(90)]
                    | "while" Exp "do" Stmt            [prec(90)]
                    | "print" "(" Exp ")" ";"          [strict]
                    | Id ":=" Exp ";"                  [strict(2) prec(40) gather (e E)]
                    | "return" Exp ";"                 [strict]
                    | "return" ";"     

      syntax Decl ::= "var" Id ";"
                    | "function" Id "(" Ids ")" Stmt

      syntax Stmts ::= Decl | Stmt
                     | Stmts Stmts [prec(100) gather(e E)]

end module

module DEMO
      imports DEMO-SYNTAX

      syntax Val ::= Int | Bool | String
                   | "lambda" "(" Ids "," Stmt ")"
      syntax Vals ::= List{Val, ","}
      syntax Exp ::= Val
      syntax KResult ::= Val

      configuration 
            <T>
                  <k> ($PGM:K ~> execute) </k>
                  <env> .Map </env>
                  <fstack> .List </fstack>

                  <nextLoc> 0 </nextLoc>
                  <store> .Map </store>
                  <genv> .Map </genv>

                  <in stream="stdin"> .List </in>
                  <out stream="stdout"> .List </out>
            </T>

      // Boolean and arithmetic expressions.
      rule B1:Bool and B2:Bool => B1 andBool B2
      rule B1:Bool or B2:Bool => B1 orBool B2
      rule not(B1:Bool) => notBool B1
      rule I1:Int + I2:Int => I1 +Int I2
      rule I1:Int * I2:Int => I1 *Int I2
      rule I1:Int - I2:Int => I1 -Int I2
      rule V1:Val == V2:Val => V1 ==K V2
      rule V1:Val != V2:Val => V1 =/=K V2

      // Variable declaration.
      syntax K ::= "undef"
      rule [var-decl]: 
            <k> var X:Id ; => . ...</k>
            <env> Env:Map => Env[L:Nat/X] </env>
            <store>... (. => L |-> undef) </store>
            <nextLoc> L => L +Int 1 </nextLoc>

      // Function declaration.
      rule [function-decl]: 
            <k> function F:Id(Xs:Ids) S:K => . ...</k>
            <env> Env:Map => Env[L:Nat/F] </env>
            <store>... (. => L |-> lambda(Xs:Ids, S)) </store>
            <nextLoc> L => L +Int 1 </nextLoc>

      // Call to main after function declarations have been added to the
      // environment.
      syntax K ::= "execute"
      rule [execute-main]:
            <k> execute => String2Id("main")(); </k>
            <env> Env:Map </env>
            <genv> _ => Env </genv>

      // Function call.
      syntax K ::=  "cc" "(" Map "," K ")"
      rule [function-call]:
            <k> _`(_`)(lambda(Xs:Ids,S:K), Vs:Vals) ~> K:K
            => bindto(Xs,Vs) ~> S ~> return; </k>
            <fstack> . => ListItem(cc(Env,K)) ...</fstack> 
            <env> Env => GEnv </env>
            <genv> GEnv:Map </genv>

      // Return from a function.
      rule [return]:
            <k> return(V:Val); ~> _ => V ~> K </k>
            <fstack> ListItem(cc(Env,K)) => . ...</fstack> 
            <env> _ => Env </env>
      rule return; => return 0; 

      // Add function parameters to its local environment.
      syntax K ::= "bindto" "(" Ids "," Vals ")"
      rule [bindto]:
            <k> bindto((X,Xs => Xs),(V,Vs:Vals => Vs)) ...</k>
            <env> Env => Env[L/X] </env>
            <store>... . => L:Nat |-> V ...</store>
            <nextLoc> L => L +Int 1 </nextLoc>
      rule [bindto-empty]:
            <k> bindto(.Ids,.Exps) => . ...</k>

      rule [sequence]:
            S1:K S2:K => S1 ~> S2

      rule [expression-statement]:
            V:Val; => .

      // The containing environment gets restored after a block. I.e.,
      // variables declared inside a block have a scope of that block.
      syntax K ::= "env" "(" Map ")"
      rule [block]: 
            <k> {Ss:Stmts} => Ss ~> env(Env:Map) ...</k>
            <env> Env </env>
      rule [block-empty]: 
            <k> {} => . ...</k>

      // Restore variable bindings after a block.
      rule [env]:
            <k> env(Env) => . ...</k>
            <env> _ => Env </env>
      
      rule [if-true]:
            if true then S:K else _ => S
      rule [if-false]:
            if false then _ else S:K => S

      rule [while]:
            <k> while B:K do S:K 
                => if B then { S while B do S } else { } ...</k>

      rule [lookup]:
            <k> X:Id => K:K ...</k>
            <env>... X |-> L ...</env>
            <store>... L |-> K ...</store>

      rule [input]:
            <k> read() => I:Int ...</k>
            <in> ListItem(I) => . ...</in> [transition]
      
      rule [output]:
            <k> print(V:Val); => . ...</k>
            <out>... . => ListItem(V) </out>

      rule [assignment]:
            <k> X:Id := V:Val; => . ...</k>
            <env>... X |-> L ...</env>
            <store>... L |-> (_ => V) ...</store>
      
end module
