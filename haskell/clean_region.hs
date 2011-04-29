{-# LANGUAGE ExistentialQuantification #-}

-- to create a region you need two things
--    1. An allocator function
--    2. A type that identifies the region
data Region elemtype rtype = Region { allocfn :: ((elemtype) -> (Ptr elemtype rtype)),
				      reg_region :: rtype }

-- a pointer is defined by two things
--    1. the element type that the pointer can reference
--    2. the identifier for the region that the pointer can point to
data Ptr elemtype rtype = Null | Ptr { address :: Integer, hackvalue :: elemtype,
					ptr_region :: rtype }


{-
 FIRST APPROACH: Here is our first approach at doing regions which doesn't quite capture everything we want
-}

-- here is a function for allocating pointers in a region
allocptr :: (Region elemtype rtype) -> elemtype -> (Ptr elemtype rtype)
allocptr r ival = (allocfn r) ival

-- here is a function for reading pointers in a function
readptr :: (Region elemtype rtype) -> (Ptr elemtype rtype) -> elemtype
readptr _ p = (hackvalue p) 

-- here is a function for writing pointers in a function
-- doesn't actually do anything
writeptr :: (Region elemtype rtype) -> (Ptr elemtype rtype) -> elemtype -> ()
writeptr _ ptr newval = ()

-- a function that returns a lambda that acts as an allocator
make_allocator rval = \ initval -> Ptr { address = 34, hackvalue = initval, ptr_region = rval }

-- create two regions, alloc pointers and try to write into them
r1 = Region { allocfn = (make_allocator "region1"), reg_region = "region1" }
r2 = Region { allocfn = (make_allocator "region2"), reg_region = "region2" }
p1 = allocptr r1 33
p2 = allocptr r2 39 
-- this write should work because it writes into the same region
z1 = writeptr r1 p1 44
-- this write should fail because it has different regions than pointers
-- but it doesn't because they aren't to the same region
z2 = writeptr r1 p1 44

{-
  SECOND APPROACH: Here is the second approach
-}

-- an AnyRegion wraps a Region by not allowing the region type to escape
-- while still ensuring that a region and all of it's elements have the
-- same type dependent on the region 
data AnyRegion elemtypecons = forall r. AnyRegion (Region (elemtypecons r) r)

-- an example of an AnyRegion 
ar1 = AnyRegion r1

{-
  LIST REGION RELATIONSHIP: create a region which contains a list of elements
-}

-- A ListElem contains an integer and a pointer to the next element in the list
data ListElem r = ListElem { value :: Integer, next :: (Ptr (ListElem r) r) }

-- We can then create a list region relationship which 
-- a list region relationship has two members
--   1. a region in which the elements are contained
--   2. a pointer to the first element in the region
data ListRR = forall r. ListRR { listrr_region :: (Region (ListElem r) r),
				 listrr_head   :: (Ptr (ListElem r) r) }

-- To make a ListRR we first have to construct a region, then we can fill in the values
make_listrr :: [Integer] -> ListRR
make_listrr v = 
  case (make_region ()) of 
      AnyRegion region -> ListRR { listrr_region = region, listrr_head = make_listrr' region v }

-- the recursive function to add elements to the region once it's been created
make_listrr' region [] = Null
make_listrr' region (x:xs) = (allocptr region (ListElem { value = x, next = make_listrr' region xs } ) )

-- a helper function to create a new region
make_region () = let
     id = "unique_region_id"
  in
     AnyRegion (Region { allocfn = (make_allocator id), reg_region = id } )

-- let's create some lists
mylist1 = make_listrr [3,4,5]
mylist2 = make_listrr [6,7,8]

-- now we write a function to sum these lists
sum_list :: ListRR -> Integer
sum_list (ListRR r1 (Ptr addr eval r2)) = sum_list' r1 (Ptr addr eval r2)

-- the recursive function for sum_list
sum_list' region Null = 0
sum_list' region ptr = let
    element = readptr region ptr
  in
    (value element) + (sum_list' region (next element))

-- here we sum up the ListRRs that we created 
total1 = sum_list mylist1
total2 = sum_list mylist2

{-
  GRAPH REGION RELATIONSHIP: Here we create a graph region relationship
-}

-- The first thing we define is a general constructor for creating a pair of
-- regions with an entangled relationship between the two regions
data RR2 tc1 tc2 = forall r1 r2. RR2 { reg1 :: (Region (tc1 r2 r1) r1),
				       reg2 :: (Region (tc2 r1 r2) r2) }

-- This is a function for making an RR2
make_rr2 = let
     id1 = "unique_region_id 1"
     id2 = "unique_region_id 2"
     r1 = Region { allocfn = (make_allocator id1), reg_region = id1 }
     r2 = Region { allocfn = (make_allocator id2), reg_region = id2 }
  in
     RR2 { reg1 = r1, reg2 = r2 }

-- Now we define GraphRR that will use RR2 to make the regions and then define
-- a pointer into the first node in the node region
data GraphRR tc1 tc2 = forall r1 r2. GraphRR { region1 :: (Region (tc1 r2 r1) r1),
					       region2 :: (Region (tc2 r1 r2) r2),
					       start :: (Ptr (tc1 r2 r1) r1) }

-- We also need to define the type constructors to be used in the graph region relationship
data Node re rn = Node { cost :: Integer, edge :: Ptr (Edge rn re) re }
data Edge rn re = Edge { dest :: Ptr (Node re rn) rn }

-- A function for creating a GraphRR using RR2
make_graphrr = let
     rr2 = make_rr2 :: RR2 Node Edge
  in
     case rr2 of
       RR2 r1 r2 -> GraphRR { region1 = r1, region2 = r2, start = Null }

