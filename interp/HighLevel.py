# High level runtime interface

def PhysicalRegion(object):
    '''This is just a wrapper object for a given RegionAllocator and RegionInstance
        for a given task.  The idea is that for a given task to run the mapper will
        provide a physical instance of the region visible to the task to be run.  We encapsulate
        the interface with a PhysicalRegion object which simply calls the corresponding
        implementation of the methods in the low-level runtime interface.'''
    def __init__(self, allocator=None, instance=None):
        self.allocator = allocator
        self.instance = instance 

    def alloc(self, count=1):
        return self.allocator.alloc(count)

    def free(self, addrs):
        self.allocator.free(addrs) 

    def read(self, address):
        return self.instance.read(address)

    def write(self, address, value):
        self.instance.write(address, value)

    def reduce(self, address, reduction_op, value):
        self.instance.reduce(address, reduction_op, value) 

def Partition(object):
    '''This is the interface for a partition object.  A base partition object can be used for
        the case of partition and then allocate and by default will be disjoint.  The specific
        sub-types of DisjointPartition and AliasedPartition will be used whenever the user
        provides a specific coloring.'''
    def __init__(self, mapper, parent, num_subregions):
        '''Initialize a new partition for a given number of sub_regions'''
        self.parent_handle = parent
        self.child_regions = []
        
        for i in range(num_subregions):
            cr = mapper.create_logical_subregion(self.parent_handle)
            self.child_regions.append(cr)

    def get_subregion(self, color):
        '''Return the handle of the logical subregion for the given color'''
        return self.child_regions[color]    

    def safe_cast(self, color, ptr):
        '''There is no coloring so we can't safe cast by default'''
        return None

    def is_disjoint(self):
        '''Test whether the partition is disjoint'''
        return True

def DisjointPartition(Partition):
    '''A partition object that explicitly notes that the coloring is disjoint'''
    def __init__(self, mapper, parent, num_subregions, color_map):
        super(mapper, parent, num_subregions) 
        self.color_map = dict(color_map)

    def get_subregion(self, color):
        super(color)

    def safe_cast(self, color, ptr):
        if (self.color_map.get(ptr) == color):
            return ptr
        else:
            return None
    
    def is_disjoint(self):
        return True

def AliasedPartition(Partition):
    '''A partition object that explicitly notes that the coloring is not disjoint'''
    def __init__(self, mapper, parent, num_subregions, color_multimap):
        super(mapper, parent, num_subregions)
        self.color_multimap = multimap(color_multimap) # Note there isn't a base multimap in python so this is fake

    def get_subregion(self, color):
        super(color)

    def safe_cast(self, color, ptr):
        if color in self.color_multimap.get(ptr):
            return ptr
        else:
            return None

    def is_disjoint(self):
        return False

def RegionRequirement(object):
    '''Specify the access requirements for the given region in a task call'''
    def __init__(self, logical_handle, access_mode, coherence_mode):
        '''Store information regarding the logical region required, its access and coherence modes'''
        self.logical_handle = logical_handle
        self.access_mode = access_mode
        self.coherence_mode = coherence_mode

def FutureValue(object):
    '''Keep track of the future return value from a task and any possible exceptions'''
    def __init__(self):
        self.result_ready = False
        self.result_value = None
        self.result_except = None

def IterationSpace(object):
    '''Define a multi-dimensional iteration space'''
    def __init__(self, num_dims, dim_sizes):
        self.num_dims = num_dims
        self.dim_sizes = dim_sizes

def TaskSpace(object):
    '''A task space is an instantiation of an iteration space for a given task'''
    def __init__(self, task_id, mapping_function, iteration_space):
        '''Will create a task space for the given task using the mapping function
            which maps points in the iteration space to arguments for the task'''
        self.task_id = task_id
        self.mapping_function = mapping_function
        self.iteration_space = iteration_space

def StaticConstraint(object):
    '''Places constraints on where tasks can be run and where the regions they
        will access must be placed'''
    def __init__(self, task_name, proc_names = None, region_map = None):
        '''For the task corresponding to a specific name provide either a set of named
            processors on which the task can be run and/or a map from region names to
            the set of possible memories the physical instances of those regions are
            allowed to be mapped to.'''
        self.task_name = task_name
        self.proc_names = proc_names
        self.region_map = region_map 

def ScheMapper(object):
    '''The schemapper is a scheduler-mapper that is responsible for providing all
        high level functionality for the region language and then mapping/scheduling
        all the operations down onto the low-level services'''
    def __init__(self, static_constraints):
        '''Initialize a new schemapper that is a singleton object for a single program.
            The argument static_constraints specifies any statically specified constraints
            that must be obeyed by the schemapper.'''
        pass

    #######################
    # Methods for regions
    #######################
    def create_logical_region(self, elmt_size):
        '''Register a new logical region with the runtime.  Return a unique handle for the logical region''' 
        pass

    def destroy_logical_region(self, handle):
        '''Destroy the logical region with the specified handle'''
        pass

    def get_physical_region(self, handle):
        '''Return the physical region associated with the handle of the given logical region'''
        pass 

    #####################
    # Methods for partitions
    #####################
    def create_disjoint_partition(self, parent, num_subregions, color_map = None):
        '''Create a disjoint partition and return the handle for the partition'''
        pass

    def create_aliased_partition(self, parent, num_subregions, color_multimap):
        '''Create an aliased partition and return the handle for the partition'''
        pass

    def get_partition(self, handle):
        '''Return a partition object corresponding to the given partition handle'''
        pass

    def destroy_partition(self, handle):
        '''Destroy the partition associated with the given handle'''
        pass

    #####################
    # Methods for task creation
    ##################### 
    def execute_task(self, task_id, region_requirements, arguments, spawn = False):
        '''Register a single task for execution with the given region requirements.  If
            spawn is true the runtime has the option of executing the task in parallel or not.  If spawn
            is true the runtime will return a future for the parent task to wait on if desired. '''
       pass

   def execute_task_set(self, task_ids, region_requirement_sets, argument_sets, futures, spawn = False, must = False):
       '''Execute a set of tasks with a given set of regions requirements, arguments, and possible futures.
           If spawn is true then the runtime has the option of running the tasks in parallel.  If spawn is
           true and must is true then the tasks must all be run in parallel.'''
       pass 

   def execute_task_space(self, task_space):
       '''Execute a task space which is just an instantiation of a given iteration space for a given task'''
       pass

