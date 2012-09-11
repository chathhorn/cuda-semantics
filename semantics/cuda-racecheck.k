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

      //                                  SId     GId     BId     TId
      syntax KResult ::= "cuda-rc-id" "(" Nat "," Nat "," Nat "," Nat ")"
      //                                   FR    FW     RC       WC    
                       | "cuda-rc-info" "(" K "," K "," Bool "," Bool ")"

      syntax K ::= "cuda-race"
      syntax K ::= "cuda-racecheck" "(" KResult ")" [strict]
      syntax K ::= "cuda-rc-update-info" "(" KResult ")" [strict]

      // TODO: should probably do some racechecking on delete.
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

      rule [cuda-rc-put-info-R-new]:
            <k> cuda-rc-put-info-R(loc(Base:Nat, Offset:Nat, 0), Info:K)
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck>
                        RaceInfo:Map (. => Offset |-> Info:K)
                  </cudaRacecheck>
            ...</object>
            when notBool (Offset in keys RaceInfo:Map)
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
      rule [cuda-rc-put-info-W-new]:
            <k> cuda-rc-put-info-W(loc(Base:Nat, Offset:Nat, 0), cuda-rc-info(FW:K, FR:K, WC:Bool, RC:Bool))
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck>
                        RaceInfo:Map (. => Offset |-> cuda-rc-info(FR:K, FW:K, RC:Bool, WC:Bool))
                  </cudaRacecheck>
            ...</object>
            when notBool (Offset in keys RaceInfo:Map)
      rule [cuda-rc-put-info-W]:
            <k> cuda-rc-put-info-W(loc(Base:Nat, Offset:Nat, 0), cuda-rc-info(FW:K, FR:K, WC:Bool, RC:Bool))
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck>... 
                        Offset |-> (_ => cuda-rc-info(FR:K, FW:K, RC:Bool, WC:Bool))
                  ...</cudaRacecheck>
            ...</object>

      rule [cuda-rc-get-info-R-new]:
            <k> cuda-rc-get-info-R(loc(Base:Nat, Offset:Nat, 0))
                  => cuda-rc-info(.K, .K, false, false)
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck> RaceInfo:Map </cudaRacecheck>
            ...</object>
            when notBool (Offset:Nat in keys RaceInfo:Map)
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
      rule [cuda-rc-get-info-W-new]:
            <k> cuda-rc-get-info-W(loc(Base:Nat, Offset:Nat, 0))
                  => cuda-rc-info(.K, .K, false, false)
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck> RaceInfo:Map </cudaRacecheck>
            ...</object>
            when notBool (Offset:Nat in keys RaceInfo:Map)
      rule [cuda-rc-get-info-W]:
            <k> cuda-rc-get-info-W(loc(Base:Nat, Offset:Nat, 0))
                  => cuda-rc-info(FW:K, FR:K, WC:Bool, RC:Bool)
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <cudaRacecheck>... 
                        Offset |-> cuda-rc-info(FR:K, FW:K, RC:Bool, WC:Bool)
                  ...</cudaRacecheck>
            ...</object>

      rule <k> cuda-rc-update-info(cuda-rc-info(.K, FX:K, Closed:Bool, CX:Bool))
                  => cuda-rc-info(ID, FX:K, Closed:Bool, CX:Bool)
            ...</k>
            <sid> SId:Nat </sid>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            where ID = cuda-rc-id(SId, GId, BId, TId)
      rule <k> cuda-rc-update-info(cuda-rc-info(ID, FX:K, Closed:Bool, CX:Bool))
                  => cuda-rc-info(ID, FX:K, Closed:Bool, CX:Bool)
            ...</k>
            <sid> SId:Nat </sid>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            where ID = cuda-rc-id(SId, GId, BId, TId)
      rule <k> cuda-rc-update-info(cuda-rc-info(You:K, FX:K, CX:Bool, Closed:Bool))
                  => cuda-rc-info(You:K, FX:K, CX:Bool, true)
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
            <k> cuda-racecheck(cuda-rc-info(FR:K, FW:K, false, _))
                  => .
            ...</k>
            <sid> SId:Nat </sid>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            when (ID ==K FW orBool FW ==K .K)
            where ID = cuda-rc-id(SId, GId, BId, TId)
      rule [cuda-racecheck-race]: 
            <k> cuda-racecheck(cuda-rc-info(FR:K, FW:K, false, _))
                  => cuda-race
            ...</k>
            <sid> SId:Nat </sid>
            <gid> GId:Nat </gid>
            <bid> BId:Nat </bid>
            <tid> TId:Nat </tid>
            when (ID =/=K FW andBool FW =/=K .K)
            where ID = cuda-rc-id(SId, GId, BId, TId)
      rule [cuda-racecheck-closed]: 
            <k> cuda-racecheck(cuda-rc-info(_, _, true, _))
                  => cuda-race
            ...</k>

      rule [cuda-rc-read-skip-loc-unknown]:
            <k> cuda-rc-read(loc(Base:Nat, Offset:Nat, _))
                  => .
            ...</k>
            <currentProgramLoc> UnknownCabsLoc </currentProgramLoc>
      rule [cuda-rc-read-skip]:
            <k> cuda-rc-read(loc(Base:Nat, Offset:Nat, _))
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            <currentProgramLoc> 
                  CabsLoc(FileName:String, Line:Int, _, _)
            </currentProgramLoc>
            <cudaDetectedRaces> Races:Set </cudaDetectedRaces>
            when (notBool (cuda-mshared in Attr:Bag) andBool notBool (cuda-mdevice in Attr:Bag))
            orBool (SetItem(FileName +String toString(Line)) in Races:Set)
      rule [cuda-rc-read-shared]:
            <k> cuda-rc-read(LOC)
                  => cuda-rc-put-info-R(LOC, cuda-rc-update-info(cuda-rc-get-info-R(LOC)))
                  ~> cuda-racecheck(cuda-rc-get-info-R(LOC))
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            <currentProgramLoc> 
                  CabsLoc(FileName:String, Line:Int, _, _)
            </currentProgramLoc>
            <cudaDetectedRaces> Races:Set </cudaDetectedRaces>
            when (cuda-mshared in Attr:Bag)
            andBool notBool (SetItem(FileName +String toString(Line)) in Races:Set)
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
            <currentProgramLoc> 
                  CabsLoc(FileName:String, Line:Int, _, _)
            </currentProgramLoc>
            <cudaDetectedRaces> Races:Set </cudaDetectedRaces>
            when notBool (cuda-mshared in Attr:Bag)
            andBool notBool (SetItem(FileName +String toString(Line)) in Races:Set)
            //andBool (cuda-mdevice in Attr:Bag)
            where LOC = loc(Base:Nat, Offset:Nat, 0)

      rule [cuda-rc-write-skip-loc-unknown]:
            <k> cuda-rc-write(loc(Base:Nat, Offset:Nat, _))
                  => .
            ...</k>
            <currentProgramLoc> UnknownCabsLoc </currentProgramLoc>
      rule [cuda-rc-write-skip]:
            <k> cuda-rc-write(loc(Base:Nat, Offset:Nat, 0))
                  => .
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            <currentProgramLoc> 
                  CabsLoc(FileName:String, Line:Int, _, _)
            </currentProgramLoc>
            <cudaDetectedRaces> Races:Set </cudaDetectedRaces>
            when (notBool (cuda-mshared in Attr:Bag) andBool notBool (cuda-mdevice in Attr:Bag))
            orBool (SetItem(FileName +String toString(Line)) in Races:Set)
      rule [cuda-rc-write-shared]:
            <k> cuda-rc-write(LOC)
                  => cuda-rc-put-info-W(LOC, cuda-rc-update-info(cuda-rc-get-info-W(LOC)))
                  ~> cuda-racecheck(cuda-rc-get-info-W(LOC))
            ...</k>
            <object>...
                  <basePtr> Base:Nat </basePtr>
                  <properties> Attr:Bag </properties>
            ...</object>
            <currentProgramLoc> 
                  CabsLoc(FileName:String, Line:Int, _, _)
            </currentProgramLoc>
            <cudaDetectedRaces> Races:Set </cudaDetectedRaces>
            when (cuda-mshared in Attr:Bag)
            andBool notBool (SetItem(FileName +String toString(Line)) in Races:Set)
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
            <currentProgramLoc> 
                  CabsLoc(FileName:String, Line:Int, _, _)
            </currentProgramLoc>
            <cudaDetectedRaces> Races:Set </cudaDetectedRaces>
            when notBool (cuda-mshared in Attr:Bag)
            andBool notBool (SetItem(FileName +String toString(Line)) in Races:Set)
            //andBool (cuda-mdevice in Attr:Bag)
            where LOC = loc(Base:Nat, Offset:Nat, 0)

      rule [cuda-race-skip-loc-unknown]:
            <k> cuda-race => . ...</k>
            <currentProgramLoc> 
                  UnknownCabsLoc
            </currentProgramLoc>
      rule [cuda-race-skip]:
            <k> cuda-race => . ...</k>
            <currentProgramLoc> 
                  CabsLoc(FileName:String, Line:Int, _, _) 
            </currentProgramLoc>
            <cudaDetectedRaces> Races:Set </cudaDetectedRaces>
            when (SetItem(FileName:String +String toString(Line:Int)) in Races:Set)
      rule [cuda-race]:
            <k> cuda-race
                  => writeToFD(1, 
                      "CUDA: Race on shared device memory detected at "
                      +String FileName:String
                      +String ":" +String toString(Line:Int)
                      //+String ":" +String toString(OffsetStart:Int)
                      //+String "-" +String toString(OffsetEnd:Int)
                      +String ".\n")
                  ~> flush(1)
            ...</k>
            <currentProgramLoc> 
                  CabsLoc(FileName:String, Line:Int, OffsetStart:Int, OffsetEnd:Int) 
            </currentProgramLoc>
            <cudaDetectedRaces> 
                  Races:Set (. => SetItem(FileName:String +String toString(Line:Int)))
            </cudaDetectedRaces>
            when notBool (SetItem(FileName +String toString(Line)) in Races)

end module
