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
                   | "read" "(" ")"
                        
      syntax Stmt ::= "{" "}"
                    | "{" Stmts "}"
                    | Exp ";"                          [strict prec(45)]
                    | "if" Exp "then" Stmt "else" Stmt [strict(1) prec(90)]
                    | "while" Exp "do" Stmt            [prec(90)]
                    | "print" "(" Exp ")" ";"          [strict]
                    | Id ":=" Exp ";"                  [strict(2) prec(40) gather(e E)]

      syntax Stmts ::= Stmt
                     | Stmts Stmts [prec(100) gather(e E)]

end module

module DEMO
      imports DEMO-SYNTAX

      syntax Val ::= Int | Bool | String
      syntax Vals ::= List{Val, ","}
      syntax Exp ::= Val
      syntax KResult ::= Val

      configuration 
            <T>
                  <k> $PGM:K </k>

                  <store> .Map </store>

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

      rule [sequence]:
            S1:K S2:K => S1 ~> S2

      rule [expression-statement]:
            V:Val; => .

      rule [block]: 
            <k> {Ss:Stmts} => Ss ...</k>
      rule [block-empty]: 
            <k> {} => . ...</k>

      rule [if-true]:
            if true then S:K else _ => S
      rule [if-false]:
            if false then _ else S:K => S

      rule [while]:
            <k> while B:K do S:K 
                => if B then { S while B do S } else { } ...</k>

      rule [lookup]:
            <k> X:Id => K:K ...</k>
            <store>... X |-> K ...</store>

      rule [input]:
            <k> read() => I:Int ...</k>
            <in> ListItem(I) => . ...</in> [transition]
      
      rule [output]:
            <k> print(V:Val); => . ...</k>
            <out>... . => ListItem(V) </out>

      rule [assignment]:
            <k> X:Id := V:Val; => . ...</k>
            <store> Sto:Map => Sto[V/X] </store>
      
end module
