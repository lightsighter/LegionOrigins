{- this is a comment? -}
foo = \x ->x+1

{- a pointer knows the type of the thing it points to, the type/value of the region it points into,
  and has an integer "address" -}
data Ptr elemtype rtype = Null | Ptr { address :: Integer, hackvalue :: elemtype, ptr_region :: rtype }

data Region elemtype rtype = Region { allocfn :: ((elemtype) -> (Ptr elemtype rtype)),
                                      reg_region :: rtype }

allocptr :: (Region elemtype rtype) -> elemtype -> (Ptr elemtype rtype)
allocptr r ival = (allocfn r) ival

readptr :: (Region elemtype rtype) -> (Ptr elemtype rtype) -> elemtype
readptr _ p = (hackvalue p)

{- not actually doing stores yet -}
writeptr :: (Region elemtype rtype) -> (Ptr elemtype rtype) -> elemtype -> ()
writeptr _ ptr newval = ()

make_allocator rval = \ initval -> Ptr { address = 34, hackvalue = initval, ptr_region = rval }

r1 = Region { allocfn = (make_allocator "region1"), reg_region = "region1" }

p1 = allocptr r1 33

z = writeptr r1 p1 44

r2 = Region { allocfn = (make_allocator "region2"), reg_region = "region2" }

p2 = allocptr r2 39

z2 = writeptr r1 p2 44 {- shouldn't work, but does -}

alloc_and_write r v = writeptr r (allocptr r v) v

data AnyRegion elemtype = forall r. AnyRegion (Region elemtype r)

data AnyRegion2 elemtypecons = forall r. AnyRegion2 (Region (elemtypecons r) r)

data AnyPtr elemtype = forall r. AnyPtr (Ptr elemtype r)

ar1 = AnyRegion r1

ar2 = AnyRegion r2

any_alloc_and_write (AnyRegion r) v = writeptr r (allocptr r v) v

data FatPtr elemtype = forall rtype. FatPtr (Region elemtype rtype) (Ptr elemtype rtype)

fat_alloc (AnyRegion r) ival = FatPtr r (allocptr r ival)

p3 = fat_alloc ar1

data ListElem r = ListElem { value :: Integer, next :: (Ptr (ListElem r) r) }

data ListRR = forall r. ListRR { listrr_region :: (Region (ListElem r) r),
                                 listrr_head ::  (Ptr (ListElem r) r) }

{-sum_list2 :: (Region (ListElem r) r) -> (Ptr (ListElem r) r) -> Integer-}
{-sum_list2 (Region _ r1) (Ptr addr eval r2) = readptr -}
sum_list2 region Null = 0
sum_list2 region ptr = let
    elem = readptr region ptr
  in
    (value elem) + (sum_list2 region (next elem))

sum_list :: ListRR -> Integer
sum_list (ListRR r1 (Ptr addr eval r2)) = sum_list2 r1 (Ptr addr eval r2)


sum_list_b :: ListRR -> Integer
sum_list_b (ListRR region head) = sum_list2 region head

sum_2_lists :: ListRR -> ListRR -> Integer
sum_2_lists (ListRR region1 head1) (ListRR region2 head2) = sum_list2 region1 head1

{- making a region is tricky - the type of the region will often depend on the region itself
   to try to get around this, instead of taking the type as an argument, we'll take a type constructor
 -}
make_region () = let
    id = "unique_region_id"
  in
    AnyRegion2 (Region { allocfn = (make_allocator id), reg_region = id })
{-
make_region :: t -> AnyRegion t
make_region v = let
    id = "unique_region_id"
  in
    AnyRegion (Region { (make_allocator v id) id)
-}
make_listrr3 region [] = Null
make_listrr3 region (x:xs) = (allocptr region (ListElem { value = x, next = make_listrr3 region xs }))
{-
make_listrr2 (AnyRegion2 region) v = ListRR { listrr_region = region,
                                           listrr_head = (allocptr region (ListElem { value = 33, next = Null })) }
-}
make_listrr2 (AnyRegion2 region) v = 
  ListRR { listrr_region = region, listrr_head = make_listrr3 region v }

make_listrr :: [Integer] -> ListRR
make_listrr v = make_listrr2 (make_region ()) v

mylist = make_listrr [3, 4, 5]
mylist2 = make_listrr [6, 7, 8]

total = sum_list mylist
{-
myptr = 
  case mylist of
    ListRR r _ -> let
        ptr = allocptr r ListElem { value = 2, next = Null }
      in
        ptr
-}
val = 
  case mylist of
    ListRR r _ -> let
        ptr1 = allocptr r ListElem { value = 2, next = Null }
        ptr2 = allocptr r ListElem { value = 2, next = ptr1 }
      in
        sum_list2 r ptr2
{-
val2 = 
  case (mylist, mylist2) of
    (ListRR r1 _, ListRR r2 _) -> let
        ptr1 = allocptr r1 ListElem { value = 2, next = Null }
        ptr2 = allocptr r2 ListElem { value = 2, next = ptr1 }
      in
        sum_list2 r2 ptr2
-}
{-
val3 = 
  case (mylist, mylist2) of
    (ListRR r1 _, ListRR r2 _) -> let
        ptr1 = allocptr r1 ListElem { value = 2, next = Null }
        ptr2 = allocptr r1 ListElem { value = 2, next = ptr1 }
      in
        sum_list2 r2 ptr2
-}
{-
        writeptr r ptr ListElem { value = 5, next = Null }
-}
{-    ListRR r _ -> writeptr (allocptr r ListElem { value = 2, next = Null }) ListElem { value = 5, next = Null }
-}
{-total2 = sum_list (listrr_region) myptr-}

{-data WackyType (t1, t2) = Wacky { a :: t1, b :: t2 }-}

data WackyType2 tc1 tc2 r1 r2 = Wacky2 { a :: tc1 r1 r2, b :: tc2 r1 r2 }

data RR2 tc1 tc2 = forall r1 r2. RR2 { reg1 :: (Region (tc1 r2 r1) r1), 
                                               reg2 :: (Region (tc2 r1 r2) r2) }


data AnyRegion3 tc = forall r1 r2. AnyRegion3 (Region (tc r1 r2) r2)

make_region2 () = let
    id = "unique_region_id"
  in
    AnyRegion3 (Region { allocfn = (make_allocator id), reg_region = id })

data Node re rn = Node { cost :: Integer, edge :: Ptr (Edge rn re) re }
data Edge rn re = Edge { dest :: Ptr (Node re rn) rn }

data GraphRR tc1 tc2 = forall r1 r2. GraphRR { region1 :: (Region (tc1 r2 r1) r1), 
                                               region2 :: (Region (tc2 r1 r2) r2),
                                               start :: (Ptr (tc1 r2 r1) r1) }
{-
make_dummy_edge :: Region (Edge r1 r2) r2 -> Ptr (Edge r1 r2) r2
make_dummy_edge r = allocptr r Edge { dest = Null }

make_dummy_node :: Region (Node r1 r2) r1 -> Integer -> Ptr (Node r1 r2) r1
make_dummy_node r v = allocptr r Node { cost = v, edge = Null }

make_graphrr2 :: (AnyRegion2 (Node re)) -> (AnyRegion2 (Edge rn)) -> Integer -> GraphRR Node Edge
make_graphrr2 (AnyRegion2 r1) (AnyRegion2 r2) v =
  GraphRR { region1 = r1, region2 = r2, start = make_dummy_node r1 v }

make_graphrr :: Integer -> GraphRR Node Edge
make_graphrr v = make_graphrr2 (make_region ()) (make_region ()) v
-}
new_graphrr v = let
    id1 = "unique_region_id 1"
    id2 = "unique_region_id 2"
    r1 = Region { allocfn = (make_allocator id1), reg_region = id1 }
    r2 = Region { allocfn = (make_allocator id2), reg_region = id2 }
  in
    GraphRR { region1 = r1, 
              region2 = r2,
              start = Null }

new_rr2 = let
    id1 = "unique_region_id 1"
    id2 = "unique_region_id 2"
    r1 = Region { allocfn = (make_allocator id1), reg_region = id1 }
    r2 = Region { allocfn = (make_allocator id2), reg_region = id2 }
  in
    RR2 { reg1 = r1, reg2 = r2 }

new_graphrr2 = let
    rr2 = new_rr2 :: RR2 Node Edge
  in
    case rr2 of
      RR2 { reg1 = r1, reg2 = r2 } -> GraphRR { region1 = r1, region2 = r2, start = Null }

