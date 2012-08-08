load cuda-headers
load dynamic-c-semantics

module CUDA-RACECHECK is
      including CUDA-THREADS
      including CUDA-RACECHECK-HEADER
      including DYNAMIC-C-SEMANTICS

      syntax K ::= "cuda-rc-read-global" "(" K "," Nat ")"
                 | "cuda-rc-read-shared" "(" K "," Nat ")"
                 | "cuda-rc-write-global" "(" K "," Nat ")"
                 | "cuda-rc-write-shared" "(" K "," Nat ")"
      syntax K ::= "cuda-rc-update-read-global" "(" K "," Nat ")"
                 | "cuda-rc-update-read-shared" "(" K "," Nat ")"
                 | "cuda-rc-update-write-global" "(" K "," Nat ")"
                 | "cuda-rc-update-write-shared" "(" K "," Nat ")"

      syntax K ::= "cuda-rc-ensure-block-exists" "(" Nat "," Nat ")"

      syntax K ::= "cuda-threadid" "(" Nat "," Nat "," Nat ")"
                 | "cuda-blockid" "(" Nat "," Nat ")"

      //                       this thread, firstRead, firstWrite, read/writeClosed, Loc
      syntax K ::= "cuda-racecheck-read" "(" K "," K "," K "," Bool "," Nat ")"
                 | "cuda-racecheck-write" "(" K "," K "," K "," Bool "," Nat ")"

      syntax K ::= "cuda-race" "(" Nat ")"

// THE FUTURE?!?
//      rule 
//            <k> racecheck-read(ThreadId, rc-tuple(.K, FirstWrite, Closed))
//                  => rc-tuple(ThreadId, FirstWrite, Closed)
//            ...</k>
//            when Closed =/=K cuda-read-closed
//      rule 
//            <k> racecheck-read(ThreadId, rc-tuple(FirstRead, FirstWrite, cuda-read-closed))
//                  => cuda-race
//            ...</k>
      
      declare cuda-racecheck-enable-shared : -> Bool
      declare cuda-racecheck-enable-global : -> Bool

      define cuda-racecheck-enable-shared => true
      define cuda-racecheck-enable-global => false
           
      rule [cuda-rc-sync-shared-skip]:
            <k> cuda-rc-sync-shared => . ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <existingBlocks> Existing:Set </existingBlocks>
            when notBool (cuda-blockid(GId, BId) in Existing)
      rule [cuda-rc-sync-shared]:
            <k> cuda-rc-sync-shared => . ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <shared>...
                  <sharedGid> GId </sharedGid>
                  <sharedBid> BId </sharedBid>
                  <firstRead> _ => . </firstRead>
                  <firstWrite> _ => . </firstWrite>
                  <readClosed> _ => . </readClosed>
                  <writeClosed> _ => . </writeClosed>
            ...</shared>

      rule [cuda-rc-sync-global]:
            <k> cuda-rc-sync-global => . ...</k>
            <globalFirstRead> _ => . </globalFirstRead>
            <globalFirstWrite> _ => . </globalFirstWrite>
            <globalReadClosed> _ => . </globalReadClosed>
            <globalWriteClosed> _ => . </globalWriteClosed>

      // A lot of copy/paste went into this creation. (TODO?)

      // *** STEP 1: cuda-rc-read/write => cuda-rc-read/write-global/shared

      rule [cuda-rc-read-skip]:
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
                  => cuda-rc-update-read-global(cuda-threadid(GId, BId, TId), loc(Base, Offset, 0))
                  ~> cuda-rc-read-global(cuda-threadid(GId, BId, TId), loc(Base, Offset, 0))
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
            andBool cuda-racecheck-enable-global
      rule <k> cuda-rc-read(loc(Base:Nat, _, _)) => . ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when notBool (cuda-mshared in Attr:Bag)
            andBool (cuda-mdevice in Attr:Bag)
            andBool notBool cuda-racecheck-enable-global

      rule [cuda-rc-read-shared]:
            <k> cuda-rc-read(loc(Base:Nat, Offset:Nat, 0))
                  => cuda-rc-ensure-block-exists(GId:Nat, BId:Nat)
                  ~> cuda-rc-update-read-shared(cuda-threadid(GId, BId, TId), loc(Base, Offset, 0))
                  ~> cuda-rc-read-shared(cuda-threadid(GId, BId, TId), loc(Base, Offset, 0))
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when cuda-mshared in Attr:Bag
            andBool cuda-racecheck-enable-shared
      rule <k> cuda-rc-read(loc(Base:Nat, _, _)) => . ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when cuda-mshared in Attr:Bag
            andBool notBool cuda-racecheck-enable-shared

      rule [cuda-rc-write-skip]:
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
                  => cuda-rc-update-write-global(cuda-threadid(GId, BId, TId), loc(Base, Offset, 0))
                  ~> cuda-rc-write-global(cuda-threadid(GId, BId, TId), loc(Base, Offset, 0))
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
            andBool cuda-racecheck-enable-global
      rule <k> cuda-rc-write(loc(Base:Nat, _, _)) => . ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when notBool (cuda-mshared in Attr:Bag)
            andBool (cuda-mdevice in Attr:Bag)
            andBool notBool cuda-racecheck-enable-global

      rule [cuda-rc-write-shared]:
            <k> cuda-rc-write(loc(Base:Nat, Offset:Nat, 0))
                  => cuda-rc-ensure-block-exists(GId, BId)
                  ~> cuda-rc-update-write-shared(cuda-threadid(GId, BId, TId), loc(Base, Offset, 0))
                  ~> cuda-rc-write-shared(cuda-threadid(GId, BId, TId), loc(Base, Offset, 0))
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when cuda-mshared in Attr:Bag
            andBool cuda-racecheck-enable-shared
      rule <k> cuda-rc-write(loc(Base:Nat, _, _)) => . ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when cuda-mshared in Attr:Bag
            andBool notBool cuda-racecheck-enable-shared

      // *** STEP 1b: cuda-rc-ensure-block-exists => . (shared only)
      
      rule [cuda-rc-ensure-block-exists]:
            <k> cuda-rc-ensure-block-exists(GId:Nat, BId:Nat)
                  => .
            ...</k>
            <existingBlocks>... SetItem(cuda-blockid(GId, BId)) ...</existingBlocks>
      rule [cuda-rc-ensure-block-exists]:
            <k> cuda-rc-ensure-block-exists(GId:Nat, BId:Nat)
                  => .
            ...</k>
            <existingBlocks> Existing:Set (. => SetItem(cuda-blockid(GId, BId))) </existingBlocks>
            (. => <shared>...
                  <sharedGid> GId </sharedGid>
                  <sharedBid> BId </sharedBid>
                  ...</shared>)
            when notBool (cuda-blockid(GId, BId) in Existing) 

      // *** STEP 2: cuda-rc-update-read/write-global/shared => .
      
      // cuda-rc-update-read-global => .
      
      rule [cuda-rc-update-read-global-first]:
            <k> cuda-rc-update-read-global(ThreadId:K, Loc:Nat)
                  => .
            ...</k>
            <globalFirstRead> FirstRead:Map (. => Loc:Nat |-> ThreadId) </globalFirstRead>
            when notBool (Loc:Nat in keys FirstRead:Map)
      rule [cuda-rc-update-read-global-already-first]:
            <k> cuda-rc-update-read-global(ThreadId:K, Loc:Nat)
                  => .
            ...</k>
            <globalFirstRead>...  Loc:Nat |-> ThreadId ...</globalFirstRead>
      rule [cuda-rc-update-read-global-close-write]:
            <k> cuda-rc-update-read-global(ThreadId:K, Loc:Nat)
                  => .
            ...</k>
            <globalFirstRead>...  Loc:Nat |-> OtherId:K ...</globalFirstRead>
            <globalWriteClosed>... (. => SetItem(Loc)) ...</globalWriteClosed>
            when OtherId =/=K ThreadId
      
      // cuda-rc-update-write-global => .

      rule [cuda-rc-update-write-global-first]:
            <k> cuda-rc-update-write-global(ThreadId:K, Loc:Nat)
                  => .
            ...</k>
            <globalFirstWrite> FirstWrite:Map (. => Loc:Nat |-> ThreadId) </globalFirstWrite>
            when notBool (Loc:Nat in keys FirstWrite:Map)
      rule [cuda-rc-update-write-global-already-first]:
            <k> cuda-rc-update-write-global(ThreadId:K, Loc:Nat)
                  => .
            ...</k>
            <globalFirstWrite>...  Loc:Nat |-> ThreadId ...</globalFirstWrite>
      rule [cuda-rc-update-write-global-close-read]:
            <k> cuda-rc-update-write-global(ThreadId:K, Loc:Nat)
                  => .
            ...</k>
            <globalFirstWrite>...  Loc:Nat |-> OtherId:K ...</globalFirstWrite>
            <globalReadClosed>... (. => SetItem(Loc)) ...</globalReadClosed>
            when OtherId =/=K ThreadId

      // Global done, now shared.
      
      // cuda-rc-update-read-shared => .
      
      rule [cuda-rc-update-read-shared-first]:
            <k> cuda-rc-update-read-shared(ThreadId:K, Loc:Nat)
                  => .
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <shared>...
                  <sharedGid> GId:Nat </sharedGid>
                  <sharedBid> BId:Nat </sharedBid>
                  <firstRead> FirstRead:Map (. => Loc:Nat |-> ThreadId) </firstRead>
            ...</shared>
            when notBool (Loc:Nat in keys FirstRead:Map)
      rule [cuda-rc-update-read-shared-already-first]:
            <k> cuda-rc-update-read-global(ThreadId:K, Loc:Nat)
                  => .
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <shared>...
                  <sharedGid> GId:Nat </sharedGid>
                  <sharedBid> BId:Nat </sharedBid>
                  <firstRead>...  Loc:Nat |-> ThreadId ...</firstRead>
            ...</shared>
      rule [cuda-rc-update-read-shared-close-write]:
            <k> cuda-rc-update-read-shared(ThreadId:K, Loc:Nat)
                  => .
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <shared>...
                  <sharedGid> GId:Nat </sharedGid>
                  <sharedBid> BId:Nat </sharedBid>
                  <firstRead>...  Loc:Nat |-> OtherId:K ...</firstRead>
                  <writeClosed>... (. => SetItem(Loc)) ...</writeClosed>
            ...</shared>
            when OtherId =/=K ThreadId
      
      // cuda-rc-update-write-shared => .

      rule [cuda-rc-update-write-shared-first]:
            <k> cuda-rc-update-write-shared(ThreadId:K, Loc:Nat)
                  => .
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <shared>...
                  <sharedGid> GId:Nat </sharedGid>
                  <sharedBid> BId:Nat </sharedBid>
                  <firstWrite> FirstWrite:Map (. => Loc:Nat |-> ThreadId) </firstWrite>
            ...</shared>
            when notBool (Loc:Nat in keys FirstWrite:Map)
      rule [cuda-rc-update-write-shared-already-first]:
            <k> cuda-rc-update-write-shared(ThreadId:K, Loc:Nat)
                  => .
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <shared>...
                  <sharedGid> GId:Nat </sharedGid>
                  <sharedBid> BId:Nat </sharedBid>
                  <firstWrite>...  Loc:Nat |-> ThreadId ...</firstWrite>
            ...</shared>
      rule [cuda-rc-update-write-shared-close-read]:
            <k> cuda-rc-update-write-shared(ThreadId:K, Loc:Nat)
                  => .
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <shared>...
                  <sharedGid> GId:Nat </sharedGid>
                  <sharedBid> BId:Nat </sharedBid>
                  <firstWrite>...  Loc:Nat |-> OtherId:K ...</firstWrite>
                  <readClosed>... (. => SetItem(Loc)) ...</readClosed>
            ...</shared>
            when OtherId =/=K ThreadId

      // *** STEP 3: cuda-rc-read/write-global/shared => cuda-racecheck-read/write

      // cuda-rc-read-global => cuda-racecheck-read
      
      rule [cuda-rc-read-global]:
            <k> cuda-rc-read-global(ThreadId:K, Loc:Nat)
                  => cuda-racecheck-read(ThreadId, ReadId, WriteId, CLOSED, Loc:Nat)
            ...</k>
            <globalFirstRead>... Loc:Nat |-> ReadId:K ...</globalFirstRead> 
            <globalFirstWrite>... Loc:Nat |-> WriteId:K ...</globalFirstWrite>
            <globalReadClosed> ReadClosed:Set </globalReadClosed>
            where CLOSED = Loc:Nat in ReadClosed:Set
      // Not possible:
      // rule [cuda-rc-read-global-no-first-read]:
      rule [cuda-rc-read-global-no-first-write]:
            <k> cuda-rc-read-global(ThreadId:K, Loc:Nat)
                  => cuda-racecheck-read(ThreadId, ReadId, .K, CLOSED, Loc:Nat)
            ...</k>
            <globalFirstRead>... Loc:Nat |-> ReadId:K ...</globalFirstRead> 
            <globalFirstWrite> FirstWrite:Map </globalFirstWrite>
            <globalReadClosed> ReadClosed:Set </globalReadClosed>
            when notBool (Loc:Nat in keys FirstWrite:Map)
            where CLOSED = Loc:Nat in ReadClosed:Set
      // Not possible:
      // rule [cuda-rc-read-global-no-first-read-write]:
      
      // cuda-rc-write-global => cuda-racecheck-write
      
      rule [cuda-rc-write-global]:
            <k> cuda-rc-write-global(ThreadId:K, Loc:Nat)
                  => cuda-racecheck-write(ThreadId, ReadId, WriteId, CLOSED, Loc:Nat)
            ...</k>
            <globalFirstRead>... Loc:Nat |-> ReadId:K ...</globalFirstRead> 
            <globalFirstWrite>... Loc:Nat |-> WriteId:K ...</globalFirstWrite>
            <globalWriteClosed> WriteClosed:Set </globalWriteClosed>
            where CLOSED = Loc:Nat in WriteClosed:Set
      // Not possible:
      // rule [cuda-rc-write-global-no-first-write]:
      rule [cuda-rc-write-global-no-first-write]:
            <k> cuda-rc-write-global(ThreadId:K, Loc:Nat)
                  => cuda-racecheck-write(ThreadId, .K, WriteId, CLOSED, Loc:Nat)
            ...</k>
            <globalFirstRead> FirstRead:Map </globalFirstRead> 
            <globalFirstWrite>... Loc:Nat |-> WriteId:K ...</globalFirstWrite>
            <globalWriteClosed> WriteClosed:Set </globalWriteClosed>
            when notBool (Loc:Nat in keys FirstRead:Map)
            where CLOSED = Loc:Nat in WriteClosed:Set
      // Not possible:
      // rule [cuda-rc-write-global-no-first-write-read]:
      
      // Global done, now shared.

      // cuda-rc-read-shared => cuda-racecheck-read
      
      rule [cuda-rc-read-shared]:
            <k> cuda-rc-read-shared(ThreadId:K, Loc:Nat)
                  => cuda-racecheck-read(ThreadId, ReadId, WriteId, CLOSED, Loc:Nat)
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <shared>...
                  <sharedGid> GId:Nat </sharedGid>
                  <sharedBid> BId:Nat </sharedBid>
                  <firstRead>... Loc:Nat |-> ReadId:K ...</firstRead> 
                  <firstWrite>... Loc:Nat |-> WriteId:K ...</firstWrite>
                  <readClosed> ReadClosed:Set </readClosed>
            ...</shared>
            where CLOSED = Loc:Nat in ReadClosed:Set
      // Not possible:
      // rule [cuda-rc-read-shared-no-first-read]:
      rule [cuda-rc-read-shared-no-first-write]:
            <k> cuda-rc-read-shared(ThreadId:K, Loc:Nat)
                  => cuda-racecheck-read(ThreadId, ReadId, .K, CLOSED, Loc:Nat)
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <shared>...
                  <sharedGid> GId:Nat </sharedGid>
                  <sharedBid> BId:Nat </sharedBid>
                  <firstRead>... Loc:Nat |-> ReadId:K ...</firstRead> 
                  <firstWrite> FirstWrite:Map </firstWrite>
                  <readClosed> ReadClosed:Set </readClosed>
            ...</shared>
            when notBool (Loc:Nat in keys FirstWrite:Map)
            where CLOSED = Loc:Nat in ReadClosed:Set
      // Not possible:
      // rule [cuda-rc-read-global-no-first-read-write]:
      
      // cuda-rc-write-shared => cuda-racecheck-write
      
      rule [cuda-rc-write-shared]:
            <k> cuda-rc-write-shared(ThreadId:K, Loc:Nat)
                  => cuda-racecheck-write(ThreadId, ReadId, WriteId, CLOSED, Loc:Nat)
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <shared>...
                  <sharedGid> GId:Nat </sharedGid>
                  <sharedBid> BId:Nat </sharedBid>
                  <firstRead>... Loc:Nat |-> ReadId:K ...</firstRead> 
                  <firstWrite>... Loc:Nat |-> WriteId:K ...</firstWrite>
                  <writeClosed> WriteClosed:Set </writeClosed>
            ...</shared>
            where CLOSED = Loc:Nat in WriteClosed:Set
      // Not possible:
      // rule [cuda-rc-write-shared-no-first-write]:
      rule [cuda-rc-write-shared-no-first-write]:
            <k> cuda-rc-write-shared(ThreadId:K, Loc:Nat)
                  => cuda-racecheck-write(ThreadId, .K, WriteId, CLOSED, Loc:Nat)
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <shared>...
                  <sharedGid> GId:Nat </sharedGid>
                  <sharedBid> BId:Nat </sharedBid>
                  <firstRead> FirstRead:Map </firstRead> 
                  <firstWrite>... Loc:Nat |-> WriteId:K ...</firstWrite>
                  <writeClosed> WriteClosed:Set </writeClosed>
            ...</shared>
            when notBool (Loc:Nat in keys FirstRead:Map)
            where CLOSED = Loc:Nat in WriteClosed:Set
      // Not possible:
      // rule [cuda-rc-write-global-no-first-write-read]:

      // *** STEP 4: cuda-racecheck-read/write => . or cuda-race

      // cuda-racecheck-read => . or cuda-race

      rule [cuda-racecheck-read-pass]:
            <k> cuda-racecheck-read(ThreadId:K, FirstRead:K, FirstWrite:K, false, Loc:Nat)
                  => .
            ...</k>
            when ThreadId ==K FirstRead
            orBool ThreadId ==K FirstWrite
            orBool FirstWrite ==K .K
      rule [cuda-racecheck-read-closed]:
            <k> cuda-racecheck-read(ThreadId:K, FirstRead:K, FirstWrite:K, true, Loc:Nat)
                  => cuda-race(Loc)
            ...</k>
      rule [cuda-racecheck-read-write]:
            <k> cuda-racecheck-read(ThreadId:K, FirstRead:K, FirstWrite:K, false, Loc:Nat)
                  => cuda-race(Loc)
            ...</k> 
            when ThreadId =/=K FirstWrite
            andBool FirstWrite =/=K .K

      // cuda-racecheck-write => . or cuda-race
      
      rule [cuda-racecheck-write-pass]:
            <k> cuda-racecheck-write(ThreadId:K, FirstRead:K, FirstWrite:K, false, Loc:Nat)
                  => .
            ...</k>
            when ThreadId ==K FirstRead
            orBool ThreadId ==K FirstWrite
            orBool FirstRead ==K .K
      rule [cuda-racecheck-write-closed]:
            <k> cuda-racecheck-write(ThreadId:K, FirstRead:K, FirstWrite:K, true, Loc:Nat)
                  => cuda-race(Loc)
            ...</k>
      rule [cuda-racecheck-write-read]:
            <k> cuda-racecheck-write(ThreadId:K, FirstRead:K, FirstWrite:K, false, Loc:Nat)
                  => cuda-race(Loc)
            ...</k>
            when ThreadId =/=K FirstRead
            andBool FirstRead =/=K .K

      rule [cuda-race]:
            <k> cuda-race(loc(Base:Nat, Offset:Nat, _))
                  => writeToFD(1, 
                        "CUDA: Race at " +String pointerToString(loc(Base:Nat, Offset:Nat, 0))
                        +String
                        #if (cuda-mshared in Attr:Bag) 
                        #then " (shared)"
                        #else " (global)"
                        #fi
                        +String " detected in thread"
                        +String " [" 
                        +String toString(GId:Nat) +String ", "
                        +String toString(BId:Nat) +String ", "
                        +String toString(TId:Nat)
                        +String "]" 
                        //+String " (" +String F:String +String ")"
                        +String ".\n")
                  ~> flush(1)
            ...</k>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            //<grids>... GId:Nat |-> cuda-grid(Call(Identifier(F:String), _), _, _, _) ...</grids>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>

end module
