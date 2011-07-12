from Runtime import Runtime, TaskContext
from Regions import *

'''These aren't required for the python version, but here are the explicit 
   types of ListElem and ListRR in a C++-like syntax:

   template <region R>
   struct ListElem {
     int           value;
     ListElem<R>@R next;  // we use @(region) instead of * for pointers, so
                          // the type of the 'next' field is a pointer that
                          // points into region R (i.e. our template parameter)
                          // and points at an element whose type is ListElem<R>
                          // (i.e. same type as us)
   }

   struct ListRR {        // ListRR is _not_ parameterized on a region...
     Region of ListElem<rlist> rlist;  // ... instead, any place in a field's
                                       // type that needs a region can use the
                                       // name of one of the fields of the same
                                       // structure (or possibly other regions
                                       // that are in scope)...
     ListElem<rlist>@rlist     head;
     int                       length;
   }

   // ... and then when you have an instance of the type, the types of its
   // fields refer to its own field contents:
   //
   //              |- foo : ListRR
   // ------------------------------------------------
   //   |- foo.rlist : Region of ListElem<foo.rlist>
   //   |- foo.head  : ListElem<foo.rlist>@foo.rlist
   ''' 
class ListElem(object):
    def __init__(self, value, next = None):
        self.value = value
        self.next = next

class ListRR(object):
    def __init__(self, rlist, head = None, length = 0):
        self.rlist = rlist
        self.head = head
        self.length = length

    def populate_list(self, *args):
        ptr = self.head
        for v in reversed(args):
            newval = ListElem(v, ptr)
            ptr = self.rlist.alloc()
            self.rlist[ptr] = newval
            self.length = self.length + 1
        self.head = ptr

    def partition_list(self, num_pieces):
        piece_length = int((self.length + num_pieces - 1) / num_pieces)
        coloring = dict()
        subheads = []
        node = self.head
        for idx in range(num_pieces):
            subheads.append(node)
            for i in range(piece_length):
                if node is None: break
                coloring[node] = idx
                node = self.rlist[node].next

        partition = Partition(self.rlist, num_pieces, coloring)

        sublists = []
        for idx in range(num_pieces):
            sublists.append(ListRR(partition.get_subregion(idx),
                                   partition.safe_cast(idx, subheads[idx]),
                                   piece_length if (idx < (num_pieces - 1)) else (self.length - piece_length * (num_pieces - 1))))
        return sublists

    def sum_list(self):
        sum = 0
        nodeptr = self.head
        for i in range(self.length):
             nodeval = self.rlist[nodeptr]
             sum = sum + nodeval.value
             nodeptr = nodeval.next
        return sum

# the 'main' function is what will be called by the simulator
# try to provide (interesting) default values for all parameters you need
def main(size = 400, num_pieces = 10):
    # create a new region
    myregion = Region("listrr", "ListElem")

    # now create a ListRR using that region (with an initially empty list)
    mylist = ListRR(myregion)

    # fill the list with the numbers from [0,size)
    mylist.populate_list(*range(size))

    # get ListRR's for the sublists
    sublists = mylist.partition_list(num_pieces)

    # run 'subtask' for each of the sublists in parallel
    subsums = map(lambda s: TaskContext.get_runtime().run_task(subtask, s),
                  sublists)

    # now total up the sums from each subtask (waiting as necessary on futures)
    total = sum(map(lambda fv: fv.get_result(), subsums))

    print "sum is %d" % total


# this subtask requires the region that is passed in via 'sublist.rlist' to be
#   readable
@region_usage(sublist__rlist = ROE)
def subtask(sublist):
    # just walk the list and return the sum of the elements
    return sublist.sum_list()

