# Low level runtime interface

import threading

class Event(object):
    '''An event is, well, an event that is expected to happen in the future.  A
	task can wait on an event, and async task/copy launches can be made to
	wait for an event to happen before they start.  Multiple events can be
	combined into a single event that doesn't occur until all sub-events
	have'''
    def __init__(self):
	'''creates an event, in the "not triggered" state'''
	self.triggered = False
	self.cond = threading.Condition()

    def trigger(self):
        '''triggers an event, waking up any tasks that are sleeping on it'''
	with self.cond:
	    if not self.triggered:
		self.triggered = True
		self.cond.notify_all()

    def wait(self):
	'''causes the current task to block until this event has occurred'''
	with self.cond:
	    if not self.triggered:
		self.cond.wait()
	
class Lock(object):
    '''A Lock is used to control access to regions, processors, etc.  It allows
	for multiple tasks to hold a lock at the same time, provided they all
	use the same 'mode', and none request exclusivity'''
    def __init__(self):
	self.mutex = threading.Lock()
	self.lock_mode = None
	self.lock_count = 0
	self.lock_excl = False
	self.pending_locks = []

    def lock(self, mode=0, exclusive=True):
	'''attempts to acquire a lock with the specified 'mode' - returns None if
	    lock is acquired, or an Event that will trigger once the lock is
	    acquired'''
	with self.mutex:
	    # if nobody has the lock, or if our lock is compatible, take it
	    if ((self.lock_mode is None) or 
	        ((self.lock_mode == mode) and not self.lock_excl and not exclusive)):
		self.lock_mode = mode
		self.lock_excl = exclusive
		self.lock_count = self.lock_count + 1
		return None

	    # otherwise, add our request to the list of pending locks and return
	    #   an event to wait on
	    pend = { 'mode': mode, 'excl': excl, 'event': Event() }
	    self.pending_locks.append(pend)
	    return pend["event"]

    def unlock(self):
	'''releases a held lock, passing the lock on to one or more pending
	    lock requests'''
	with self.mutex:
	    # simple case 1: multiple locks are still held - just dec count
	    if self.lock_count > 1:
		self.lock_count = self.lock_count - 1
		return

	    # simple case 2: this was last held lock, but nothing is pending
	    if len(self.pending_locks) == 0:
		self.lock_mode = None
		self.lock_count = 0
		return

	    # complicated case 3: one or more pending requests exist - choose
	    #  first one and all others compatible with it
            pend = self.pending_locks.pop(0)
	    self.lock_mode = pend["mode"]
	    self.lock_excl = pend["excl"]
	    self.lock_count = 1
            pend["event"].trigger()

	    if not self.lock_excl:
		leftovers = []
		for p in self.pending_locks:
		    if (p["mode"] == self.lock_mode) and not p["excl"]:
			self.lock_count = self.lock_count + 1
			p["event"].trigger()
		    else:
			leftovers.append(p)
		self.pending_locks = leftovers

class RegionMetaData(object):
    '''A RegionMetaData is used to track the physical region instances and
	allocators that correspond to the same logical region.  It knows which
	are the master allocators and instances, and allows control over that
	masterness to be claimed (i.e. locked) by individual tasks'''

    def __init__(self, size, master_location):
	'''creates a new RegionMetaData that holds at least 'size' entries and 
	    creates a master allocator and master instance in the specified
	    location'''
	self.master_allocator = RegionAllocator(self, master_location)
	self.master_allocator.free(range(size)) # hack to make all addresses initially available

	self.master_instance = RegionInstance(self, master_location)

	self.lock = Lock()

class RegionAllocator(object):
    '''A RegionAllocator is something that understands (at least partially) which
	addresses within a region are in use and supports the allocation of 
	currently-unused addresses and the free'ing of currently-used addresses.
	A given RegionAllocator lives in the same memory location for its entire
	lifetime, but it has the ability to synchronize with other RegionAllocators
	to support alloc/free in multiple locations for the same region'''

    def __init__(self, region, memory_location):
	'''creates a new RegionAllocator for 'region' in 'memory_location'.  It is
	    NOT automatically synchronized - use the appropriate copy's (at
	    the appropriate time)'''
	self.region = region
	self.memory_location = memory_location
	self.avail_addresses = []
	self.num_avail = 0
        self.mutex = threading.Lock()
	self.lock = Lock()

    def alloc(self, count=1):
	'''allocates one or more currently-unused addresses and returns them to
	    the caller.  Returns 'None' if enough addresses aren't available.'''
	with self.mutex:
	    if count > self.num_avail: return None
	    addrs = self.avail_addresses[0:count]
	    self.avail_addresses = self.avail_addresses[count:]
            self.num_avail = self.num_avail - count
            return addrs

    def free(self, *addrs):
	'''returns one or more addresses to the available address pool'''
	with self.mutex:
	    self.avail_addresses.extend(*addrs)
	    self.num_avail = self.num_avail + len(*addrs)

class RegionInstance(object):
    '''A RegionInstance is a copy of the actual bits stored in a region.  Values
	can be read, written, or reduced using addresses assigned by a
	RegionAllocator.  When multiple instances exist for a single region, they
	are not automatically kept coherent - you'll need to copy bits around
	manually'''

    def __init__(self, region, memory_location):
	'''creates a new RegionInstance for 'region' in the specified memory
            location (which cannot change for the lifetime of the instance)'''
	self.region = region
	self.memory_location = memory_location
	self.lock = Lock()
	self.mutex = threading.Lock() # for internal use only!
        self.contents = dict()

    def read(self, address):
	return self.contents.get(address)

    def write(self, address, value):
	with self.mutex:
	    self.contents[address] = value

    def reduce(self, address, reduction_op, value):
	with self.mutex:
	    self.contents[address] = reduction_op(self.contents.get(address), value)

    def copy_to(self, to_inst):
	'''copies all the contents of the current region istance to 'to_inst',
            overwriting whatever was there - the copy way be done asynchronously,
	    so an event is returned that will trigger when the copy is completed'''
	with (self.mutex, to_inst.mutex):
	    ev = Event()
	    to_inst.contents.update(from_inst.contents)
	    # copy is blocking in this implementation so trigger event right away
	    ev.trigger()
	    return ev

class Processor(object):
    '''A Processor is a location at which code can be run.  It keeps a queue of
	pending tasks.  It is guaranteed to be able to run at least one task
	(independently of what other processors are doing), but does not guarantee
	to be able to run more than one task, even if the first task is waiting
	on an event'''

    def __init__(self):
	'''creates a new processor, with an initially empty task list'''
	self.active_task = None
	self.pending_tasks = []
	self.mutex = threading.Lock()
	self.cond = threading.Condition(self.mutex)
	self.worker = threading.Thread(target = self.worker_loop)
	self.worker.daemon = True
	self.worker.start()

    def spawn(self, func, args=None, wait_for=None):
	'''adds a new task to the processor's task queue - returns an event that
	    will be triggered when that task is completed'''
	t = { "func": func, "args": args, "wait_for": wait_for, "event": Event() }
	with self.mutex:
	    self.pending_tasks.append(t)
	    self.cond.notify_all()  # TODO: do this much more gracefully
	return t["event"]

    def worker_loop(self):
	while True:
	    # first step - get something to run
	    with self.mutex:
		while len(self.pending_tasks) == 0:
		    self.cond.wait()
		self.active_task = self.pending_tasks.pop(0)

	    if self.active_task["wait_for"] is not None:
		self.active_task["wait_for"].wait()

	    if self.active_task["func"] is not None:
		self.active_task["func"](*self.active_task["args"])

	    if self.active_task["event"] is not None:
		self.active_task["event"].trigger()

	    self.active_task = None
