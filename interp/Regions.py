import inspect
import re

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

    def is_subset_of(self, other):
        # can't read or write if our parent can't
        if self.is_reader and not other.is_reader: return False
        if self.is_writer and not other.is_writer: return False

        # a reduction (if specified) must match exactly
        if (self.reduction_op is not None) and (self.reduction_op <> other.reduction_op): return False
        
        # transitive subset relations are A < E, S < E, R < S
        if self.exclusivity == self.EXCL:
            return (other.exclusivity == self.EXCL)

        if self.exclusivity == ATOMIC:
            return (other.exclusivity in (EXCL, ATOMIC))

        if self.exclusivity == SIMULT:
            return (other.exclusivity in (EXCL, SIMULT))

        if self.exclusivity == RELAXED:
            return (other.exclusivity in (EXCL, SIMULT, RELAXED))

        raise UnreachableCode
             

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


class NoRegionUsageException(Exception):
    def __init__(self, func):
        self.func = func

    def __str__(self):
        return "Function '%s' must be decorated with @region_usage if you want to call it as a task" % (self.func.func_name);


def get_task_regions(f, args, kwargs):
    '''combines a function's region_usage info with the actual arguments to a
       task call to determine which regions are needed by a given task instance'''
    if not hasattr(f, "region_usage"):
        raise NoRegionUsageException(f)

    # if the "function" is actually a bound method, pull out the 'self' value and 
    #   prepend it to the arg list
    if hasattr(f, "im_self") and (f.im_self is not None):
        print type(args)
        args = [ f.im_self ] + list(args)

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

    @classmethod
    def from_str(self, s):
        print s, type(s)
        m = re.match(r'^(?P<addr>\d+)\@(?P<region>.+)$', s)
        if m is None:
            raise BadPointerString(s)
        return Pointer(Region.name_to_region(m.group('region')), int(m.group('address')))

############################################################

def __unique_pointer__(start):
    i = start
    while True:
       i = i + 1
       yield i

class RegionInstance(object):
    # HACK: for now, all values in all regions are stored in one big ptr->value
    #   map.  This lets us solve the aliased-data-stores-between-parents-and-children
    #   problem later.
    global_store = dict()
    unique_pointers = __unique_pointer__(0)

    def __init__(self, region, location):
        self.region = region
        self.location = location
        #self.store = dict()

    def __repr__(self):
        return repr(self.region) + "(" + self.location + ")"

    def alloc(self):
        addr = RegionInstance.unique_pointers.next()
        ptr = Pointer(self.region, addr)
        self.region.ptrs[addr] = self.region

        # add pointer to all parents of our region as well
        for r in self.region.all_supersets():
            r.ptrs[addr] = self.region

        #self.store[addr] = None
        RegionInstance.global_store[addr] = None
        return ptr

    def readptr(self, ptr):
        # TODO: figure out how to do this dynamic check - exact match is too strict
        #if (ptr.region != self.region):
        #    raise RegionMismatchException(ptr, self)
        if ptr.address not in self.region.ptrs: #self.store:
            print str(self.region.ptrs)
            raise UnknownPointerException(ptr, self)
        #return self.store[ptr.address]
        return RegionInstance.global_store[ptr.address]

    def writeptr(self, ptr, newval):
        # TODO: figure out how to do this dynamic check - exact match is too strict
        #if (ptr.region != self.region):
        #    raise RegionMismatchException(ptr, self)
        if ptr.address not in self.region.ptrs: #self.store:
            raise UnknownPointerException(ptr, self)
        #self.store[ptr.address] = newval
        RegionInstance.global_store[ptr.address] = newval

    def reduceptr(self, ptr, redval, reduce_op):
        # TODO: figure out how to do this dynamic check - exact match is too strict
        #if (ptr.region != self.region):
        #    raise RegionMismatchException(ptr, self)
        if ptr.address not in self.region.ptrs: #self.store:
            raise UnknownPointerException(ptr, self)
        #origval = self.store[ptr.address]
        origval = RegionInstance.global_store[ptr.address]
        #self.store[ptr.address] = reduce_op(origval, redval)
        RegionInstance.global_store[ptr.address] = reduce_op(origval, redval)

############################################################

class Region(object):
    unique_names = dict()

    def _get_unique_name(self, base_name):
        if base_name not in Region.unique_names:
            Region.unique_names[base_name] = self
            return base_name

        name = base_name
        if re.search(r'\(\d+\)$', base_name) is None: name += "(1)"
        while True:
            name = re.sub(r'\(\d+\)$', lambda m: "("+str(eval(m.group(0)+"+1"))+")", name)
            if name not in Region.unique_names: break
        Region.unique_names[name] = self
        return name

    @classmethod
    def name_to_region(self, name):
        return self.unique_names[name]

    def __init__(self, my_name, elem_type, auto_bind = True):
        self.name = self._get_unique_name(my_name)
        self.elem_type = elem_type
        self.instances = set()
        self.partitions = set()
        self.supersets = set()
        self.ptrs = dict()
        if auto_bind:
            from Runtime import TaskContext, RegionBinding
            self.master = self.get_instance(TaskContext.get_processor())
            TaskContext.get_current_context().add_region_bindings(RegionBinding(self, self.master, RWE))
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

    def is_subset_of(self, region, transitive = True):
        if region in self.supersets: return True
        if not transitive:           return False
        if region == self:           return True
        for s in self.supersets:
            if s.is_subset_of(region): return True
        return False

    def all_supersets(self):
        '''returns an iterator that walks through all supersets of this region, visiting each
           region only once'''
        seen = set()
        todo = list(self.supersets)
        while len(todo) > 0:
            n = todo.pop()
            if n not in seen:
                seen.add(n)
                todo.extend(n.supersets)
                yield n

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
        region.partitions.add(self)       

        for i in range(num_subregions):
            cr = Region(region.name + "[" + str(i) + "]", region.elem_type, auto_bind = False)
            self.child_regions.append(cr)
            cr.supersets.add(region)

        if color_map == None:
            self.color_map = dict()
        else:
            self.color_map = dict(color_map)
            for p, v in color_map.iteritems():
                #print k
                #p = Pointer.from_str(k)
                #print k, repr(p), v
                self.child_regions[v].ptrs[p.address] = region.ptrs[p.address]

    def get_subregion(self, index):
        return self.child_regions[index]

    def safe_cast(self, index, ptr):
        if (self.color_map.get(ptr) == index):
            return ptr
        else:
            return None

############################################################

