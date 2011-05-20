import threading
from Regions import Region
from Tasks import TaskThread, FutureValue

class Runtime(object):
    def __init__(self, machine_desc):
        self.machine_desc = machine_desc

    def run_application(self, main_func, *main_args, **main_kwargs):
        '''start the main thread for an application on an arbitrary processor'''
        main = TaskThread(main_func, main_args, main_kwargs,
                          TaskContext(self, None, "foo"))
        main.start()
        return main.result.get_result()

    def run_task(self, task_func, *task_args, **task_kwargs):
        '''spawn a new task, specifying which regions are needed, with optional mapping hints'''
        # see which regions the task needs
        from Regions import get_task_regions
        regions_needed = get_task_regions(task_func, task_args, task_kwargs)
        # TODO: actual mapping
        location = "foo"
        newctx = TaskContext(self, None, location)

        for r in regions_needed:
            # TODO: check that the requested capabilities are a subset of what we have
            newctx.add_region_bindings(RegionBinding(r["region"],
                                                     r["region"].get_instance(location),
                                                     r["mode"]))
        newthr = TaskThread(task_func, task_args, task_kwargs, newctx)

        newthr.start()
        # in debug mode, wait for child thread to finish
        do_debug = True
        if do_debug:
            newthr.join()
            newthr.result.get_result()  # if the subtask had an exception, so will this
        return newthr.result  # this is a future, not the actual result

############################################################

class RegionBinding(object):
    def __init__(self, logical_region, phys_inst, mode):
        self.logical_region = logical_region
        self.phys_inst = phys_inst
        self.mode = mode


class UnmappedRegionException(Exception):
    def __init__(self, context, logical_region):
        self.context = context
        self.logical_region = logical_region

    def __str__(self):
        return "Region (" + str(self.logical_region) + ") unmapped in (" + str(self.context.bindings) + ")"

############################################################


class TaskContext(object):
    thread_local_storage = threading.local()

    def __init__(self, runtime, task, processor, bindings = None):
        self.runtime = runtime
        self.task = task
        self.processor = processor
        self.bindings = bindings if bindings is not None else []
        pass

    @classmethod
    def get_current_context(self): return TaskContext.thread_local_storage.context

    @classmethod
    def set_current_context(self, new_ctx): TaskContext.thread_local_storage.context = new_ctx
    
    # all of the methods below are defined as class methods - ideally they'd be
    #   both class and instance methods so that we can detect erroneous cases
    #   in which the wrong (i.e. not yours) context is used
    @classmethod
    def get_runtime(self):
        '''returns a pointer to the global runtime object'''
        self = self.get_current_context()
        return self.runtime

    @classmethod
    def get_task(self):
        '''returns the current task'''
        self = self.get_current_context()
        return self.task

    @classmethod
    def get_processor(self):
        '''returns the processor the task is running on'''
        self = self.get_current_context()
        return self.processor

    # not classmethod
    def add_region_bindings(self, *args):
        '''adds region bindings to the current context'''
        #self = self.get_current_context()
        self.bindings.append(*args)

    @classmethod
    def get_region_binding(self, logical_region, exact = True, must_match = True):
        '''returns the region binding being used for a logical region'''
        self = self.get_current_context()

        # try for exact matches first (not sure this matters)
        for b in self.bindings:
            if b.logical_region == logical_region: return b

        # now allow bindings of supersets of the region we want
        for b in self.bindings:
            if logical_region.is_subset_of(b.logical_region):
                return RegionBinding(logical_region, b.phys_inst, b.mode)

        if must_match:
            raise UnmappedRegionException(self, logical_region)
        return None


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

