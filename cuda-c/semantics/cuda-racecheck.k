load cuda-headers
load dynamic-c-semantics

module CUDA-RACECHECK is
      including CUDA-THREADS
      including CUDA-RACECHECK-HEADER
      including DYNAMIC-C-SEMANTICS

      ops   cuda-rc-read-m-aux
            cuda-rc-write-m-aux
            : Nat Bool -> K

      macro cuda-rc-read-m(Loc:Nat) = cuda-rc-read-m-aux(Loc:Nat, cuda-cfg-raceChecking)

      macro cuda-rc-read-m-aux(Loc:Nat, true) = cuda-rc-read(Loc:Nat)
      macro cuda-rc-read-m-aux(Loc:Nat, false) = .K

      macro cuda-rc-write-m(Loc:Nat) = cuda-rc-write-m-aux(Loc:Nat, cuda-cfg-raceChecking)

      macro cuda-rc-write-m-aux(Loc:Nat, true) = cuda-rc-write(Loc:Nat)
      macro cuda-rc-write-m-aux(Loc:Nat, false) = .K

      syntax K ::= "cuda-race" "(" Nat ")"

      //                                  SId     GId     BId     TId
      syntax KResult ::= "cuda-rc-id" "(" Nat "," Nat "," Nat "," Nat ")"
      //                                   FR    FW     RC       WC       race
                       | "cuda-rc-info" "(" K "," K "," Bool "," Bool "," Bool ")"

      syntax K ::= "cuda-rc-new-info" "(" Nat ")"
                 | "cuda-race" "(" Nat ")"
      syntax K ::= "cuda-racecheck" "(" Nat "," KResult ")" [strict(2)]
      syntax K ::= "cuda-rc-update-info" "(" KResult ")" [strict]


      // TODO: should probably do some racechecking on delete. It should count
      // as a write to every offset in the deleted structure.
      rule [cuda-rc-on-delete-shared]:
            <k> cuda-rc-on-delete(Loc:Nat, Attrs:Bag, RC:Map)
                  => .
            ...</k>
            <cudaSharedObjects>... ListItem(Loc) => . ...</cudaSharedObjects>
            <cudaDeviceObjects>... ListItem(Loc) => . ...</cudaDeviceObjects>
            when (cuda-mshared in Attrs)
      rule [cuda-rc-on-delete-device-only]:
            <k> cuda-rc-on-delete(Loc:Nat, Attrs:Bag, RC:Map)
                  => .
            ...</k>
            <cudaDeviceObjects>... ListItem(Loc) => . ...</cudaDeviceObjects>
            when (cuda-mdevice in Attrs)
            andBool notBool (cuda-mshared in Attrs)
      rule [cuda-rc-on-delete-host]:
            <k> cuda-rc-on-delete(Loc:Nat, Attrs:Bag, RC:Map)
                  => .
            ...</k>
            when notBool (cuda-mdevice in Attrs)
            andBool notBool (cuda-mshared in Attrs)

      // *** Syncing.
      
      syntax K ::= "cuda-rc-sync" "(" List ")"
      syntax K ::= "cuda-rc-sync-aux" "(" Nat ")"
      rule [cuda-rc-sync-shared-skip]:
            <k> cuda-rc-sync-shared 
                  => cuda-rc-sync(SharedObjects:List) 
            ...</k>
            <cudaSharedObjects> SharedObjects:List </cudaSharedObjects>
           
      //    rule [cuda-rc-sync-global]:
      //          <k> cuda-rc-sync-global 
      //                => cuda-rc-sync(DeviceObjects:List) 
      //          ...</k>
      //          <cudaDeviceObjects> DeviceObjects:List </cudaDeviceObjects>
      // TODO: Skipping global for now.
      rule [cuda-rc-sync-global]:
            <k> cuda-rc-sync-global 
                  => .
            ...</k>

      rule [cuda-rc-sync]:
            <k> (. => cuda-rc-sync-aux(Loc:Nat))
                  ~> cuda-rc-sync((ListItem(Loc:Nat) => .) Locs:List)
            ...</k>
      rule [cuda-rc-sync-done]:
            <k> cuda-rc-sync(.List)
                  => .
            ...</k>

      rule [cuda-rc-sync-aux]:
            <k> cuda-rc-sync-aux(loc(Base:Nat, 0, 0))
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck> (_ => .Map)  </cudaRacecheck>
            ...</object>

      // *** Race checking.
      
      syntax K ::= "cuda-rc-put-info-R" "(" Nat "," KResult ")" [strict(2)]
                 | "cuda-rc-put-info-W" "(" Nat "," KResult ")" [strict(2)]
                 | "cuda-rc-get-info-R" "(" KResult ")"
                 | "cuda-rc-get-info-W" "(" KResult ")"

      rule [cuda-rc-put-info-R]:
            <k> cuda-rc-put-info-R(loc(Base:Nat, Offset:Nat, 0), Info:K)
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck>... 
                        Offset |-> (_ => Info:K)
                  ...</cudaRacecheck>
            ...</object>
      rule [cuda-rc-put-info-W]:
            <k> cuda-rc-put-info-W(loc(Base:Nat, Offset:Nat, 0), cuda-rc-info(FW:K, FR:K, WC:Bool, RC:Bool, Race:Bool))
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck>... 
                        Offset |-> (_ => cuda-rc-info(FR:K, FW:K, RC:Bool, WC:Bool, Race:Bool))
                  ...</cudaRacecheck>
            ...</object>

      rule [cuda-rc-get-info-R]:
            <k> cuda-rc-get-info-R(loc(Base:Nat, Offset:Nat, 0))
                  => Info:K
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck>... 
                        Offset |-> Info:K
                  ...</cudaRacecheck>
            ...</object>
      rule [cuda-rc-get-info-W]:
            <k> cuda-rc-get-info-W(loc(Base:Nat, Offset:Nat, 0))
                  => cuda-rc-info(FW:K, FR:K, WC:Bool, RC:Bool, Race:Bool)
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck>... 
                        Offset |-> cuda-rc-info(FR:K, FW:K, RC:Bool, WC:Bool, Race:Bool)
                  ...</cudaRacecheck>
            ...</object>

      rule <k> cuda-rc-update-info(cuda-rc-info(.K, FX:K, Closed:Bool, CX:Bool, Race:Bool))
                  => cuda-rc-info(ID, FX:K, Closed:Bool, CX:Bool, Race:Bool)
            ...</k>
            <sid> SId:Nat </sid>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            where ID = cuda-rc-id(SId, GId, BId, TId)
      rule <k> cuda-rc-update-info(cuda-rc-info(ID, FX:K, Closed:Bool, CX:Bool, Race:Bool))
                  => cuda-rc-info(ID, FX:K, Closed:Bool, CX:Bool, Race:Bool)
            ...</k>
            <sid> SId:Nat </sid>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            where ID = cuda-rc-id(SId, GId, BId, TId)
      rule <k> cuda-rc-update-info(cuda-rc-info(You:K, FX:K, Closed:Bool, CX:Bool, Race:Bool))
                  => cuda-rc-info(You:K, FX:K, true, CX:Bool, Race:Bool)
            ...</k>
            <sid> SId:Nat </sid>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            when ID =/=K You:K
            andBool You:K =/=K .K
            where ID = cuda-rc-id(SId, GId, BId, TId)

      // Detects a race for the "read" case, but we just twiddle the tuple and
      // use it again for the "write" case. Same deal as above
      // (cuda-rc-update-info).

      rule [cuda-racecheck-skip]:
            <k> cuda-racecheck(_, cuda-rc-info(FR:K, FW:K, false, _, _))
                  => .
            ...</k>
            <sid> SId:Nat </sid>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            when (ID ==K FW orBool FW ==K .K)
            where ID = cuda-rc-id(SId, GId, BId, TId)
      rule [cuda-racecheck-race]: 
            <k> cuda-racecheck(Loc:K, cuda-rc-info(FR:K, FW:K, false, _, _))
                  => cuda-race(Loc)
            ...</k>
            <sid> SId:Nat </sid>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            when (ID =/=K FW andBool FW =/=K .K)
            where ID = cuda-rc-id(SId, GId, BId, TId)
      rule [cuda-racecheck-closed]: 
            <k> cuda-racecheck(Loc:K, cuda-rc-info(_, _, true, _, _))
                  => cuda-race(Loc)
            ...</k>

      rule [cuda-rc-read-skip]:
            <k> cuda-rc-read(loc(Base:Nat, Offset:Nat, _))
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when notBool (cuda-mshared in Attr:Bag)
            andBool notBool (cuda-mdevice in Attr:Bag)
      rule [cuda-rc-read-shared]:
            <k> cuda-rc-read(LOC)
                  => cuda-rc-new-info(LOC)
                  ~> cuda-rc-put-info-R(LOC, cuda-rc-update-info(cuda-rc-get-info-R(LOC)))
                  ~> cuda-racecheck(LOC, cuda-rc-get-info-R(LOC))
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when (cuda-mshared in Attr:Bag)
            where LOC = loc(Base:Nat, Offset:Nat, 0)
      // Skipping global for now.
      rule [cuda-rc-read-global]:
            <k> cuda-rc-read(LOC)
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when notBool (cuda-mshared in Attr:Bag)
            //andBool (cuda-mdevice in Attr:Bag)
            where LOC = loc(Base:Nat, Offset:Nat, 0)

      rule [cuda-rc-write-skip]:
            <k> cuda-rc-write(loc(Base:Nat, Offset:Nat, 0))
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when notBool (cuda-mshared in Attr:Bag)
            andBool notBool (cuda-mdevice in Attr:Bag)
      rule [cuda-rc-write-shared]:
            <k> cuda-rc-write(LOC)
                  => cuda-rc-new-info(LOC)
                  ~> cuda-rc-put-info-W(LOC, cuda-rc-update-info(cuda-rc-get-info-W(LOC)))
                  ~> cuda-racecheck(LOC, cuda-rc-get-info-W(LOC))
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when (cuda-mshared in Attr:Bag)
            where LOC = loc(Base:Nat, Offset:Nat, 0)
      // Skipping global for now.
      rule [cuda-rc-write-global]:
            <k> cuda-rc-write(LOC)
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            when notBool (cuda-mshared in Attr:Bag)
            //andBool (cuda-mdevice in Attr:Bag)
            where LOC = loc(Base:Nat, Offset:Nat, 0)

      rule [cuda-rc-new-info-skip]:
            <k> cuda-rc-new-info(loc(Base:Nat, Offset:Nat, _))
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck>... Offset |-> _ ...</cudaRacecheck>
            ...</object>
      rule [cuda-rc-new-info]:
            <k> cuda-rc-new-info(loc(Base:Nat, Offset:Nat, _))
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck> Info:Map (. => Offset |-> cuda-rc-info(.K, .K, false, false, false)) </cudaRacecheck>
            ...</object>
            when notBool (Offset:Nat in keys Info:Map)

      // Skip printing if this location has already been raced.
      rule [cuda-race-skip]:
            <k> cuda-race(LOC)
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck>... 
                        Offset |-> cuda-rc-info(_, _, _, _, true) 
                  ...</cudaRacecheck>
            ...</object>
            where LOC = loc(Base:Nat, Offset:Nat, 0)
      rule [cuda-race]:
            <k> cuda-race(LOC)
                  => writeToFD(1, 
                        "CUDA: Race at " +String pointerToString(LOC)
                        +String
                        #if (cuda-mshared in Attr:Bag) 
                        #then " (shared)"
                        #else " (global)"
                        #fi
                        +String " detected in thread"
                        +String " [" 
                        +String toString(SId:Nat) +String ", "
                        +String toString(GId:Nat) +String ", "
                        +String toString(BId:Nat) +String ", "
                        +String toString(TId:Nat)
                        +String "]" 
                        //+String " (" +String F:String +String ")"
                        +String ".\n")
                  ~> flush(1)
            ...</k>
            <sid> SId:Nat </sid>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
                  <cudaRacecheck>... 
                        Offset |-> cuda-rc-info(_, _, _, _, (false => true)) 
                  ...</cudaRacecheck>
            ...</object>
            where LOC = loc(Base:Nat, Offset:Nat, 0)

end module
