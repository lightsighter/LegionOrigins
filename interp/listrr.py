import Runtime
import Tasks
import Regions

class ListElem(object):
    def __init__(self, value, next = None):
        self.value = value
        self.next = next

class ListRR(object):
    def __init__(self, region, head = None, length = 0):
        self.region = region
        self.head = head
        self.length = length

    def populate_list(self, value_list):
        ptr = self.head
        for v in reversed(values):
            newval = ListElem(v, ptr)
            ptr = self.region.alloc()
            self.region[ptr] = newval
            self.length = self.length + 1
        self.head = ptr

    def partition_list(self, num_pieces):
        piece_length = int(self.length + num_pieces - 1 / num_pieces)
        coloring = dict()
        subheads = []
        node = self.head
        for idx in range(num_pieces):
            subheads.add(node)
            for i in range(piece_length):
                if node is None: break
                coloring[node] = idx
                node = self.region[node].next

        partition = Partition(self.region, num_pieces, coloring)

        sublists = []
        for idx in range(num_pieces):
            sublists.add(ListRR(partition.get_subregion(idx),
                                partition.safe_cast(idx, subheads[idx]),
                                piece_length if (idx < (num_pieces - 1)) else (self.length - piece_length * (num_pieces - 1))))
        return sublists

    def sum_list(self):
        sum = 0
        nodeptr = self.head
        for i in range(self.length):
             nodeval = self.region[nodeptr]
             sum = sum + nodeval.value
             nodeptr = nodeval.next
        return sum


def list_example(size, num_pieces):
    values = [ i for i in range(size) ]
    myregion = Region("listrr", "ListElem")
    mylist = ListRR(myregion)
    mylist.populate_list(values)
    sublists = mylist.partition_list(num_pieces)
    runtime = Context.get_runtime()
    subsums = [ runtime.run_task(subtask, (s)) for s in sublists ]  # parallel!
    total = 0
    for s in subsums:
        total = total + s.get_result()
    print "sum is " + total

def subtask(sublist):
    return sublist.sum_list()

if __name__ == "__main__":
    Runtime.run_application(list_example, (400, 10))

