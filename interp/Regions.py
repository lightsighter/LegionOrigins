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

