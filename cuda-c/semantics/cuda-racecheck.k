load cuda-headers

module CUDA-RACECHECK is
      including CUDA-RACECHECK-HEADER

      syntax K ::= "cuda-rc-read-global" "(" K "," Nat ")"
                 | "cuda-rc-read-shared" "(" K "," Nat ")"
                 | "cuda-rc-write-global" "(" K "," Nat ")"
                 | "cuda-rc-write-shared" "(" K "," Nat ")"
      syntax K ::= "cuda-rc-update-read-global" "(" K "," Nat ")"
                 | "cuda-rc-update-read-shared" "(" K "," Nat ")"
                 | "cuda-rc-update-write-global" "(" K "," Nat ")"
                 | "cuda-rc-update-write-shared" "(" K "," Nat ")"

      syntax K ::= "cuda-rc-ensure-block-exists" "(" Nat "," Nat ")"

      syntax K ::= "cuda-thread-id" "(" Nat "," Nat "," Nat ")"
                 | "cuda-block-id" "(" Nat "," Nat ")"

      //                       this thread, firstRead, firstWrite, read/writeClosed, Loc
      syntax K ::= "cuda-racecheck-read" "(" K "," K "," K "," Bool "," Nat ")"
                 | "cuda-racecheck-write" "(" K "," K "," K "," Bool "," Nat ")"

      syntax K ::= "cuda-race" "(" K "," K "," Nat ")"

      rule [cuda-rc-sync-shared-skip]:
            <k> cuda-rc-sync-shared => . ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaRaceChecking>...
                  <existingBlocks> Existing:Set </existingBlocks>
            ...</cudaRaceChecking>
            when notBool (cuda-block-id(GId, BId) in Existing)
      rule [cuda-rc-sync-shared]:
            <k> cuda-rc-sync-shared => . ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaRaceChecking>...
                  <shared>...
                        <gid> GId </gid>
                        <bid> BId </bid>
                        <firstRead> _ => . </firstRead>
                        <firstWrite> _ => . </firstWrite>
                        <readClosed> _ => . </readClosed>
                        <writeClosed> _ => . </writeClosed>
                  ...</shared>
            ...</cudaRaceChecking>
            when notBool (cuda-block-id(GId, BId) in Existing)

      rule [cuda-rc-sync-global]:
            <k> cuda-rc-sync-global => . ...</k>
            <cudaRaceChecking>...
                  <firstRead> _ => . </firstRead>
                  <firstWrite> _ => . </firstWrite>
                  <readClosed> _ => . </readClosed>
                  <writeClosed> _ => . </writeClosed>
            ...</cudaRaceChecking>

      // A lot of copy/paste went into this creation. (TODO?)

      // *** STEP 1: cuda-rc-read/write => cuda-rc-read/write-global/shared

      rule [cuda-rc-read-not-shared]:
            <k> cuda-rc-read(loc(Base:Nat, _, 0)) // the "0" is the bitoffset
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when notBool (cuda-mshared in Attr:Bag)
            andBool notBool (cuda-mdevice in Attr:Bag)

      rule [cuda-rc-read-global]:
            <k> cuda-rc-read(loc(Base:Nat, Offset:Nat, 0))
                  => cuda-rc-update-read-global(cuda-thread-id(GId, BId, TId), loc(Base, Offset, 0))
                  ~> cuda-rc-read-global(cuda-thread-id(GId, BId, TId), loc(Base, Offset, 0))
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when notBool (cuda-mshared in Attr:Bag)
            andBool (cuda-mdevice in Attr:Bag)

      rule [cuda-rc-read-shared]:
            <k> cuda-rc-read(loc(Base:Nat, Offset:Nat, 0))
                  => cuda-rc-ensure-block-exists(GId:Nat, BId:Nat)
                  ~> cuda-rc-update-read-shared(cuda-thread-id(GId, BId, TId), loc(Base, Offset, 0))
                  ~> cuda-rc-read-shared(cuda-thread-id(GId, BId, TId), loc(Base, Offset, 0))
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when cuda-mshared in Attr:Bag

      rule [cuda-rc-write-not-shared]:
            <k> cuda-rc-write(loc(Base:Nat, _, 0)) // the "0" is the bitoffset
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when notBool (cuda-mshared in Attr:Bag)
            andBool notBool (cuda-mdevice in Attr:Bag)

      rule [cuda-rc-write-global]:
            <k> cuda-rc-write(loc(Base:Nat, Offset:Nat, 0))
                  => cuda-rc-update-write-global(cuda-thread-id(GId, BId, TId), loc(Base, Offset, 0))
                  ~> cuda-rc-write-global(cuda-thread-id(GId, BId, TId), loc(Base, Offset, 0))
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when notBool (cuda-mshared in Attr:Bag)
            andBool (cuda-mdevice in Attr:Bag)

      rule [cuda-rc-write-shared]:
            <k> cuda-rc-write(loc(Base:Nat, Offset:Nat, 0))
                  => cuda-rc-ensure-block-exists(GId, BId)
                  ~> cuda-rc-update-write-shared(cuda-thread-id(GId, BId, TId), loc(Base, Offset, 0))
                  ~> cuda-rc-write-shared(cuda-thread-id(GId, BId, TId), loc(Base, Offset, 0))
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when cuda-mshared in Attr:Bag

      // *** STEP 1b: cuda-rc-ensure-block-exists => . (shared only)
      
      rule [cuda-rc-ensure-block-exists]:
            <k> cuda-rc-ensure-block-exists(GId:Nat, BId:Nat)
                  => .
            ...</k>
            <cudaRaceChecking>...
                  <existingBlocks>... SetItem(cuda-block-id(GId, BId)) ...</existingBlocks>
            ...</cudaRaceChecking>
      rule [cuda-rc-ensure-block-exists]:
            <k> cuda-rc-ensure-block-exists(GId:Nat, BId:Nat)
                  => .
            ...</k>
            <cudaRaceChecking>...
                  <existingBlocks> Existing:Set (. => SetItem(cuda-block-id(GId, BId))) </existingBlocks>
                  (. => <shared>...
                        <gid> GId </gid>
                        <bid> BId </bid>
                        ...</shared>)
            ...</cudaRaceChecking>
            when notBool (cuda-block-id(GId, BId) in Existing) 

      // *** STEP 2: cuda-rc-update-read/write-global/shared => .
      
      // cuda-rc-update-read-global => .
      
      rule [cuda-rc-update-read-global-first]:
            <k> cuda-rc-update-read-global(Id:K, Loc:Nat)
                  => .
            ...</k>
            <cudaRaceChecking>...
                  <firstRead> FirstRead:Map (. => Loc |-> Id) </firstRead>
            ...</cudaRaceChecking>
            when notBool (Loc in keys FirstRead)
      rule [cuda-rc-update-read-global-already-first]:
            <k> cuda-rc-update-read-global(Id:K, Loc:Nat)
                  => .
            ...</k>
            <cudaRaceChecking>...
                  <firstRead>...  Loc |-> Id ...</firstRead>
            ...</cudaRaceChecking>
      rule [cuda-rc-update-read-global-close-write]:
            <k> cuda-rc-update-read-global(Id:K, Loc:Nat)
                  => .
            ...</k>
            <cudaRaceChecking>...
                  <firstRead>...  Loc |-> OtherId:K ...</firstRead>
                  <writeClosed>... (. => SetItem(Loc)) ...</writeClosed>
            ...</cudaRaceChecking>
            when OtherId =/=K Id
      
      // cuda-rc-update-write-global => .

      rule [cuda-rc-update-write-global-first]:
            <k> cuda-rc-update-write-global(Id:K, Loc:Nat)
                  => .
            ...</k>
            <cudaRaceChecking>...
                  <firstWrite> FirstWrite:Map (. => Loc |-> Id) </firstWrite>
            ...</cudaRaceChecking>
            when notBool (Loc in keys FirstWrite)
      rule [cuda-rc-update-write-global-already-first]:
            <k> cuda-rc-update-write-global(Id:K, Loc:Nat)
                  => .
            ...</k>
            <cudaRaceChecking>...
                  <firstWrite>...  Loc |-> Id ...</firstWrite>
            ...</cudaRaceChecking>
      rule [cuda-rc-update-write-global-close-read]:
            <k> cuda-rc-update-write-global(Id:K, Loc:Nat)
                  => .
            ...</k>
            <cudaRaceChecking>...
                  <firstWrite>...  Loc |-> OtherId:K ...</firstWrite>
                  <readClosed>... (. => SetItem(Loc)) ...</readClosed>
            ...</cudaRaceChecking>
            when OtherId =/=K Id

      // Global done, now shared.
      
      // cuda-rc-update-read-shared => .
      
      rule [cuda-rc-update-read-shared-first]:
            <k> cuda-rc-update-read-shared(Id:K, Loc:Nat)
                  => .
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaRaceChecking>...
                  <shared>...
                        <gid> GId:Nat </gid>
                        <bid> BId:Nat </bid>
                        <firstRead> FirstRead:Map (. => Loc |-> Id) </firstRead>
                  ...</shared>
            ...</cudaRaceChecking>
            when notBool (Loc in keys FirstRead)
      rule [cuda-rc-update-read-shared-already-first]:
            <k> cuda-rc-update-read-global(Id:K, Loc:Nat)
                  => .
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaRaceChecking>...
                  <shared>...
                        <gid> GId:Nat </gid>
                        <bid> BId:Nat </bid>
                        <firstRead>...  Loc |-> Id ...</firstRead>
                  ...</shared>
            ...</cudaRaceChecking>
      rule [cuda-rc-update-read-shared-close-write]:
            <k> cuda-rc-update-read-shared(Id:K, Loc:Nat)
                  => .
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaRaceChecking>...
                  <shared>...
                        <gid> GId:Nat </gid>
                        <bid> BId:Nat </bid>
                        <firstRead>...  Loc |-> OtherId:K ...</firstRead>
                        <writeClosed>... (. => SetItem(Loc)) ...</writeClosed>
                  ...</shared>
            ...</cudaRaceChecking>
            when OtherId =/=K Id
      
      // cuda-rc-update-write-shared => .

      rule [cuda-rc-update-write-shared-first]:
            <k> cuda-rc-update-write-shared(Id:K, Loc:Nat)
                  => .
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaRaceChecking>...
                  <shared>...
                        <gid> GId:Nat </gid>
                        <bid> BId:Nat </bid>
                        <firstWrite> FirstWrite:Map (. => Loc |-> Id) </firstWrite>
                  ...</shared>
            ...</cudaRaceChecking>
            when notBool (Loc in keys FirstWrite)
      rule [cuda-rc-update-write-shared-already-first]:
            <k> cuda-rc-update-write-shared(Id:K, Loc:Nat)
                  => .
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaRaceChecking>...
                  <shared>...
                        <gid> GId:Nat </gid>
                        <bid> BId:Nat </bid>
                        <firstWrite>...  Loc |-> Id ...</firstWrite>
                  ...</shared>
            ...</cudaRaceChecking>
      rule [cuda-rc-update-write-shared-close-read]:
            <k> cuda-rc-update-write-shared(Id:K, Loc:Nat)
                  => .
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaRaceChecking>...
                  <shared>...
                        <gid> GId:Nat </gid>
                        <bid> BId:Nat </bid>
                        <firstWrite>...  Loc |-> OtherId:K ...</firstWrite>
                        <readClosed>... (. => SetItem(Loc)) ...</readClosed>
                  ...</shared>
            ...</cudaRaceChecking>
            when OtherId =/=K Id

      // *** STEP 3: cuda-rc-read/write-global/shared => cuda-racecheck-read/write

      // cuda-rc-read-global => cuda-racecheck-read
      
      rule [cuda-rc-read-global]:
            <k> cuda-rc-read-global(Id:K, Loc:Nat)
                  => cuda-racecheck-read(Id, ReadId, WriteId, Loc in ReadClosed)
            ...</k>
            <cudaRaceChecking>...
                 <firstRead>... Loc |-> ReadId:K ...</firstRead> 
                 <firstWrite>... Loc |-> WriteId:K ...</firstWrite>
                 <readClosed> ReadClosed:Set </readClosed>
            ...</cudaRaceChecking>
      // Not possible:
      // rule [cuda-rc-read-global-no-first-read]:
      rule [cuda-rc-read-global-no-first-write]:
            <k> cuda-rc-read-global(Id:K, Loc:Nat)
                  => cuda-racecheck-read(Id, ReadId, .K, Loc in ReadClosed)
            ...</k>
            <cudaRaceChecking>...
                 <firstRead>... Loc |-> ReadId:K ...</firstRead> 
                 <firstWrite> FirstWrite:Map </firstWrite>
                 <readClosed> ReadClosed:Set </readClosed>
            ...</cudaRaceChecking>
            when notBool (Loc in keys FirstWrite)
      // Not possible:
      // rule [cuda-rc-read-global-no-first-read-write]:
      
      // cuda-rc-write-global => cuda-racecheck-write
      
      rule [cuda-rc-write-global]:
            <k> cuda-rc-write-global(Id:K, Loc:Nat)
                  => cuda-racecheck-write(Id, ReadId, WriteId, Loc in WriteClosed)
            ...</k>
            <cudaRaceChecking>...
                 <firstRead>... Loc |-> ReadId:K ...</firstRead> 
                 <firstWrite>... Loc |-> WriteId:K ...</firstWrite>
                 <writeClosed> WriteClosed:Set </writeClosed>
            ...</cudaRaceChecking>
      // Not possible:
      // rule [cuda-rc-write-global-no-first-write]:
      rule [cuda-rc-write-global-no-first-write]:
            <k> cuda-rc-write-global(Id:K, Loc:Nat)
                  => cuda-racecheck-write(Id, .K, WriteId, Loc in WriteClosed)
            ...</k>
            <cudaRaceChecking>...
                 <firstRead> FirstRead:Map </firstRead> 
                 <firstWrite>... Loc |-> WriteId:K ...</firstWrite>
                 <writeClosed> WriteClosed:Set </writeClosed>
            ...</cudaRaceChecking>
            when notBool (Loc in keys FirstRead)
      // Not possible:
      // rule [cuda-rc-write-global-no-first-write-read]:
      
      // Global done, now shared.

      // cuda-rc-read-shared => cuda-racecheck-read
      
      rule [cuda-rc-read-shared]:
            <k> cuda-rc-read-shared(Id:K, Loc:Nat)
                  => cuda-racecheck-read(Id, ReadId, WriteId, Loc in ReadClosed)
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaRaceChecking>...
                  <shared>...
                        <gid> GId:Nat </gid>
                        <bid> BId:Nat </bid>
                        <firstRead>... Loc |-> ReadId:K ...</firstRead> 
                        <firstWrite>... Loc |-> WriteId:K ...</firstWrite>
                        <readClosed> ReadClosed:Set </readClosed>
                  ...</shared>
            ...</cudaRaceChecking>
      // Not possible:
      // rule [cuda-rc-read-shared-no-first-read]:
      rule [cuda-rc-read-shared-no-first-write]:
            <k> cuda-rc-read-shared(Id:K, Loc:Nat)
                  => cuda-racecheck-read(Id, ReadId, .K, Loc in ReadClosed)
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaRaceChecking>...
                  <shared>...
                        <gid> GId:Nat </gid>
                        <bid> BId:Nat </bid>
                        <firstRead>... Loc |-> ReadId:K ...</firstRead> 
                        <firstWrite> FirstWrite:Map </firstWrite>
                        <readClosed> ReadClosed:Set </readClosed>
                  ...</shared>
            ...</cudaRaceChecking>
            when notBool (Loc in keys FirstWrite)
      // Not possible:
      // rule [cuda-rc-read-global-no-first-read-write]:
      
      // cuda-rc-write-shared => cuda-racecheck-write
      
      rule [cuda-rc-write-shared]:
            <k> cuda-rc-write-shared(Id:K, Loc:Nat)
                  => cuda-racecheck-write(Id, ReadId, WriteId, Loc in WriteClosed)
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaRaceChecking>...
                  <shared>...
                        <gid> GId:Nat </gid>
                        <bid> BId:Nat </bid>
                        <firstRead>... Loc |-> ReadId:K ...</firstRead> 
                        <firstWrite>... Loc |-> WriteId:K ...</firstWrite>
                        <writeClosed> WriteClosed:Set </writeClosed>
                  ...</shared>
            ...</cudaRaceChecking>
      // Not possible:
      // rule [cuda-rc-write-shared-no-first-write]:
      rule [cuda-rc-write-shared-no-first-write]:
            <k> cuda-rc-write-shared(Id:K, Loc:Nat)
                  => cuda-racecheck-write(Id, .K, WriteId, Loc in WriteClosed)
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <cudaRaceChecking>...
                  <shared>...
                        <gid> GId:Nat </gid>
                        <bid> BId:Nat </bid>
                        <firstRead> FirstRead:Map </firstRead> 
                        <firstWrite>... Loc |-> WriteId:K ...</firstWrite>
                        <writeClosed> WriteClosed:Set </writeClosed>
                  ...</shared>
            ...</cudaRaceChecking>
            when notBool (Loc in keys FirstRead)
      // Not possible:
      // rule [cuda-rc-write-global-no-first-write-read]:

      // *** STEP 4: cuda-racecheck-read/write => . or cuda-race

      // cuda-racecheck-read => . or cuda-race
      
      rule [cuda-racecheck-read-pass]:
            <k> cuda-racecheck-read(Id:K, FirstRead:K, FirstWrite:K, false, Loc:Nat)
                  => .
            </k>
            when Id ==K FirstRead 
            orBool Id ==K FirstWrite 
            orBool FirstWrite ==K .K
      rule [cuda-racecheck-read-closed]:
            <k> cuda-racecheck-read(Id:K, FirstRead:K, FirstWrite:K, true, Loc:Nat)
                  => cuda-race(Id, FirstWrite, Loc)
            </k>
      rule [cuda-racecheck-read-write]:
            <k> cuda-racecheck-read(Id:K, FirstRead:K, FirstWrite:K, false, Loc:Nat)
                  => cuda-race(Id, FirstWrite, Loc)
            </k>
            when Id =/=K FirstWrite 
            andBool FirstWrite =/=K .K

      // cuda-racecheck-write => . or cuda-race
      
      rule [cuda-racecheck-write-pass]:
            <k> cuda-racecheck-write(Id:K, FirstRead:K, FirstWrite:K, false, Loc:Nat)
                  => .
            </k>
            when Id ==K FirstRead 
            orBool Id ==K FirstWrite 
            orBool FirstRead ==K .K
      rule [cuda-racecheck-write-closed]:
            <k> cuda-racecheck-write(Id:K, FirstRead:K, FirstWrite:K, true, Loc:Nat)
                  => cuda-race(Id, FirstWrite, Loc)
            </k>
      rule [cuda-racecheck-write-read]:
            <k> cuda-racecheck-write(Id:K, FirstRead:K, FirstWrite:K, false, Loc:Nat)
                  => cuda-race(Id, FirstWrite, Loc)
            </k>
            when Id =/=K FirstRead 
            andBool FirstRead =/=K .K



end module
