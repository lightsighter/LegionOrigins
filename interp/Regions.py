import inspect

############################################################

# Region access modes

class RegionAccessMode(object):
    EXCL, ATOMIC, SIMULT, RELAXED = range(4)

    def __init__(self, is_reader, is_writer, reduction_op = None, exclusivity = EXCL):
        self.is_reader = is_reader
        self.is_writer = is_writer
        self.reduction_op = reduction_op
        self.exclusivity = exclusivity

    def __repr__(self):
        s = ("RW" if self.is_writer else "RO") if self.is_reader else "Rd"
        if self.reduction_op is not None:
            s = s + "(" + self.reduction_op.func_name + ")"
        s = s + ('EASR'[self.exclusivity])
        return s
             

ROE = RegionAccessMode(True, False, None, RegionAccessMode.EXCL)
ROA = RegionAccessMode(True, False, None, RegionAccessMode.ATOMIC)
ROS = RegionAccessMode(True, False, None, RegionAccessMode.SIMULT)
ROR = RegionAccessMode(True, False, None, RegionAccessMode.RELAXED)

RWE = RegionAccessMode(True, True, None, RegionAccessMode.EXCL)
RWA = RegionAccessMode(True, True, None, RegionAccessMode.ATOMIC)
RWS = RegionAccessMode(True, True, None, RegionAccessMode.SIMULT)
RWR = RegionAccessMode(True, True, None, RegionAccessMode.RELAXED)

RdE = lambda op: RegionAccessMode(False, True, op, RegionAccessMode.EXCL)
RdA = lambda op: RegionAccessMode(False, True, op, RegionAccessMode.ATOMIC)
RdS = lambda op: RegionAccessMode(False, True, op, RegionAccessMode.SIMULT)
RdR = lambda op: RegionAccessMode(False, True, op, RegionAccessMode.RELAXED)

############################################################

# Decorator for functions to specify their region usage (as references to
#   actual parameters at task invocation time)

def region_usage(*args, **kwargs):
    '''Decorator for functions to specify their region usage (as references to
       actual parameters at task invocation time)'''

    def wrap_fn(f):
        f.region_options = list(args)
        f.region_usage = dict(**kwargs)
        f.argspec = inspect.getargspec(f)
        return f
    return wrap_fn


def _match_args(argspec, args, kwargs):
    '''helper function that matches positional and keyword arguments in a function
       call to the formal parameter names'''

    match = dict()
    # positional arguments match to the function's arg list
    for (n, v) in zip(argspec.args, args):
        match[n] = v

    # now handle keyword=value pairs
    for (n, v) in kwargs.iteritems():
        if n in match:
            raise DuplicateArgumentError(n)
        match[n] = v

    # finally, handle defaults
    reqd_args = len(argspec.args)
    if argspec.defaults is not None:
        reqd_args = reqd_args - len(argspec.defaults)

        for i in range(len(argspec.defaults)):
            if argspec.args[i + reqd_args] not in match:
                match[argspec.args[i + reqd_args]] = argspec.defaults[i]

    # check that all required arguments did end up with values
    for i in range(reqd_args):
        if argspec.args[i] not in match:
            raise MissingArgumentError(argspec.args[i])


    return match


def get_task_regions(f, args, kwargs):
    '''combines a function's region_usage info with the actual arguments to a
       task call to determine which regions are needed by a given task instance'''

    match = _match_args(f.argspec, args, kwargs)
    usage = []
    for (rn, rv) in f.region_usage.iteritems():
        curr = match
        for field in rn.split("__"):
            # case 1: field refers to an object attribute (a.k.a. field)
            if hasattr(curr, field):
                curr = getattr(curr, field)
                continue

            # case 2: field refers to a dictionary key
            if type(curr) is dict:
                curr = curr[field]
                continue

            # TODO: handle things like arrays/tuples?
            raise FieldExtractionException(curr, field)

        # TODO: check that we ended at something that's a logical region
        usage.append(dict(region = curr, mode = rv))
    return usage

#def call_func(f, *args, **kwargs):
#    if not hasattr(f, "region_usage"):
#        raise NoRegionUsageException(f)
#    print "calling: " + f.__name__
#    print "args: " + str(args)
#    print "kwargs: " + str(kwargs)
#    print str(match_args(f.argspec, args, kwargs))
#    print str(get_task_regions(f, args, kwargs))
#    return f(*args, **kwargs)


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
    def __init__(self, my_name, elem_type, auto_bind = True):
        self.name = my_name
        self.elem_type = elem_type
        self.instances = set()
        if auto_bind:
            from Runtime import TaskContext, RegionBinding
            self.master = self.get_instance(TaskContext.get_processor())
            TaskContext.add_region_bindings(RegionBinding(self, self.master, RWE))
        else:
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

    def alloc(self):
        '''helper function that simply looks up the right physical instance
           in the caller's context and performs the alloc on that'''
        from Runtime import TaskContext
        binding = TaskContext.get_region_binding(self)
        return binding.phys_inst.alloc()

    def free(self):
        '''helper function that simply looks up the right physical instance
           in the caller's context and performs the free on that'''
        from Runtime import TaskContext
        binding = TaskContext.get_region_binding(self)
        return binding.phys_inst.free()

    def readptr(self, ptr):
        '''helper function that simply looks up the right physical instance
           in the caller's context and performs the read on that'''
        from Runtime import TaskContext
        binding = TaskContext.get_region_binding(self)
        return binding.phys_inst.readptr(ptr)

    def writeptr(self, ptr, newval):
        '''helper function that simply looks up the right physical instance
           in the caller's context and performs the store on that'''
        from Runtime import TaskContext
        binding = TaskContext.get_region_binding(self)
        binding.phys_inst.writeptr(ptr, newval)

    def reduceptr(self, ptr, redval):
        '''helper function that simply looks up the right physical instance
           in the caller's context and performs the reduction on that'''
        from Runtime import TaskContext
        binding = TaskContext.get_region_binding(self)
        binding.phys_inst.reduceptr(ptr, redval, binding.reduce_op)

    '''for even more helperness, make region[ptr] syntax work for reads and writes'''
    __getitem__ = readptr
    __setitem__ = writeptr


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
            cr = Region(region.name + "[" + str(i) + "]", region.elem_type, auto_bind = False)
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

