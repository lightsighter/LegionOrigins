import threading

class Runtime(object):
    def __init__(self, machine_desc):
        self.machine_desc = machine_desc

    def run_application(self, main_func, main_args):
        '''start the main thread for an application on an arbitrary processor'''
        pass

    def run_task(self, task_func, task_args, regions_needed, mapping_hints = None):
        '''spawn a new task, specifying which regions are needed, with optional mapping hints'''
        pass

    def create_region(self, name, elem_type):
        pass

############################################################

class Context(object):
    def __init__(self):
        current_context = threading.local()

    def get_runtime(self):
        '''returns a pointer to the global runtime object'''
        return self.runtime

    def get_task(self):
        '''returns the current task'''
        return self.task

    def get_processor(self):
        '''returns the processor the task is running on'''
        return self.processor

    def get_region_instance(self, logical_region):
        '''returns the region instance being used for a logical region'''
        if logical_region not in self.instances:
            raise UnmappedRegionException(self, logical_region)
        return self.instances[logical_region]

    def get_region_reduction_op(self, logical_region):
        '''returns the reduction op being used for a logical region'''
        if logical_region not in self.reductions:
            raise UnknownRegionReductionException(self, logical_region)
        return self.reductions[logical_region]

    def readptr(self, logical_region, ptr):
        '''helper function to read a pointer using a logical region'''
        return self.get_region_instance(logical_region).readptr(ptr)

    def writeptr(self, logical_region, ptr, newval):
        '''helper function to write a pointer using a logical region'''
        self.get_region_instance(logical_region).writeptr(ptr, newval)

    def reduceptr(self, logical_region, ptr, redval):
        '''helper function to reduce to a pointer using a logical region (and the implied reduction op)'''
        op = self.get_region_reduction_op(logical_region)
        self.get_region_instance(logical_region).reduceptr(ptr, redval, op)

############################################################

class UnknownPointerException(Exception):
    def __init__(self, pointer, instance):
        self.pointer = pointer
        self.instance = instance

    def __str__(self):
        return "Pointer " + repr(self.pointer) + " not known to region instance " + repr(self.instance)

############################################################

class Pointer(object):
    def __init__(self, region, address):
        self.region = region
        self.address = address

    def __repr__(self):
        return repr(self.address) + "@" + repr(self.region)

############################################################

def __unique_pointer__():
    i = 0
    while 1:
       i = i + 1
       yield i

class RegionInstance(object):
    def __init__(self, region, location):
        self.region = region
        self.location = location
        self.store = dict()

    def __repr__(self):
        return repr(self.region) + "(" + self.location + ")"

    def alloc(self):
        addr = __unique_pointer__()
        ptr = Pointer(self.region, addr)
        self.store[addr] = None
        return ptr

    def readptr(self, ptr):
        if (ptr.region != self.region):
            raise RegionMismatchException(ptr, self)
        if ptr.address not in self.store:
            raise UnknownPointerException(ptr, self)
        return self.store[ptr.address]

    def writeptr(self, ptr, newval):
        if (ptr.region != self.region):
            raise RegionMismatchException(ptr, self)
        if ptr.address not in self.store:
            raise UnknownPointerException(ptr, self)
        self.store[ptr.address] = newval

    def reduceptr(self, ptr, redval, reduce_op):
        if (ptr.region != self.region):
            raise RegionMismatchException(ptr, self)
        if ptr.address not in self.store:
            raise UnknownPointerException(ptr, self)
        origval = self.store[ptr.address]
        self.store[ptr.address] = reduce_op(origval, redval)

############################################################

class Region(object):
    def __init__(self, my_name, elem_type):
        self.name = my_name
        self.elem_type = elem_type
        self.instances = set()
        self.master = None

    def __repr__(self):
        return "R:" + self.name + ":" + self.elem_type

    def get_instance(self, location):
        # find an existing instance in the right location and return it,
        for i in [i for i in self.instances if i.location == location]:
            return i
        # or create a new one
        i = RegionInstance(self, location)
        self.instances.add(i)
        return i

    def set_master(self, inst):
        if inst <> self.master:
            if self.master <> None:
               inst.store = self.master.store.copy()
            self.master = inst

############################################################

class Partition(object):
    def __init__(self, region, num_subregions, color_map = None):
        self.parent_region = region
        self.child_regions = []
        if color_map == None:
            self.color_map = dict()
        else:
            self.color_map = dict(color_map)
        for i in range(num_subregions):
            cr = Region(region.name + "[" + i + "]")
            cr.ptrs = dict((k, v) for k, v in region.ptrs.iteritems() if (color_map.get(k) == i))
            self.child_regions.add(cr)

    def get_subregion(self, index):
        return self.child_regions[index]

    def safe_cast(self, index, ptr):
        if (self.color_map.get(ptr) == index):
            return ptr
        else:
            return None

############################################################

def make_listrr(values = None):
    r = Region("listrr", "ListElem")
    ri = r.get_instance("some_location")
    r.set_master(ri)
    ptr = None
    for v in values.__reversed__():
        newptr = ri.alloc()
        newval = {'value': v, 'next': ptr}
        ri.writeptr(newptr, newval)
        ptr = newptr
    return {'rl': r, 'head': ptr}

def sum_listrr(listrr):
    ri = listrr["rl"].get_instance("some_other_location")
    listrr["rl"].set_master(ri)
    ptr = listrr["head"]
    total = 0
    while ptr <> None:
        elem = ri.readptr(ptr)
        total = total + elem["value"]
        ptr = elem["next"]
    return total

if __name__ == "__main__":
    mylist = make_listrr([1, 2, 3, 4, 5])

    mysum = sum_listrr(mylist)

    print mysum

