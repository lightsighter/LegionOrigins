
#include "lowlevel.h"

#include <cstdio>
#include <cstring>
#include <cassert>
#include <cstdlib>

#include <map>
#include <set>
#include <list>
#include <vector>

#include <pthread.h>

#define BASE_EVENTS	64	
#define BASE_LOCKS	64	

// Minimum number of tasks are processor can
// have in its queues before the scheduler is invoked
#define MIN_SCHED_TASKS	4

// The number of threads for this version
#define NUM_PROCS	4
// Maximum memory in global in bytes
#define MAX_GLOBAL_MEM 	67108864
#define MAX_LOCAL_MEM	32768

#ifdef DEBUG_LOW_LEVEL
#define PTHREAD_SAFE_CALL(cmd)			\
	{					\
		int ret = (cmd);		\
		if (ret != 0) {			\
			fprintf(stderr,"PTHREAD error: %s = %d (%s)\n", #cmd, ret, strerror(ret));	\
			exit(1);		\
		}				\
	}
#else
#define PTHREAD_SAFE_CALL(cmd)			\
	(cmd);
#endif

namespace RegionRuntime {
  namespace LowLevel {
    // Implementation for each of the runtime objects
    class EventImpl;
    class LockImpl;
    class MemoryImpl;
    class ProcessorImpl;
    class RegionMetaDataImpl;
    class RegionAllocatorImpl;
    class RegionInstanceImpl;

    class Runtime {
    public:
      Runtime(Machine *m);
    public:
      static Runtime* get_runtime(void) { return runtime; } 

      EventImpl*           get_event_impl(Event e);
      LockImpl*            get_lock_impl(Lock l);
      MemoryImpl*          get_memory_impl(Memory m);
      ProcessorImpl*       get_processor_impl(Processor p);
      RegionMetaDataImpl*  get_metadata_impl(RegionMetaDataUntyped m);
      RegionAllocatorImpl* get_allocator_impl(RegionAllocatorUntyped a);
      RegionInstanceImpl*  get_instance_impl(RegionInstanceUntyped i);

      EventImpl*           get_free_event(void);
      LockImpl*            get_free_lock(void);
      RegionMetaDataImpl*  get_free_metadata(Memory m, size_t num_elmts, size_t elmt_size);
    protected:
      static Runtime *runtime;
    protected:
      friend class Machine;
      std::vector<EventImpl*> events;
      std::vector<LockImpl*> locks;
      std::vector<MemoryImpl*> memories;
      std::vector<ProcessorImpl*> processors;
      std::vector<RegionMetaDataImpl*> metadatas;
      std::vector<RegionAllocatorImpl*> allocators;
      std::vector<RegionInstanceImpl*> instances;
      Machine *machine;
      pthread_mutex_t mutex;
    };

    /* static */
    Runtime *Runtime::runtime = NULL;

    __thread unsigned local_proc_id;

    // Any object which can be triggered should be able to triggered
    // This will include Events and Locks
    class Triggerable {
    public:
	virtual void trigger(void) = 0;
    };

    
    ////////////////////////////////////////////////////////
    // Events 
    ////////////////////////////////////////////////////////

    class EventImpl : public Triggerable {
    public:
	typedef unsigned EventIndex;
	typedef unsigned EventGeneration;
	static Event::ID make_id(EventIndex index, EventGeneration gen) {
	  return ((((unsigned long long)index) << 32) | gen);
	}
	static EventIndex get_index(Event::ID id) {
	  return (id >> 32);
	}
	static EventGeneration get_gen(Event::ID id) {
	  return (id & 0xFFFFFFFFULL);
	}
    public:
	EventImpl(EventIndex idx, bool activate=false) {
	  index = idx;
	  in_use = activate;
	  generation = 0;
	  PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
	  PTHREAD_SAFE_CALL(pthread_cond_init(&wait_cond,NULL));
	}
	
	// test whether an event has triggered without waiting
	bool has_triggered(EventGeneration needed_gen);
	// block until event has triggered
	void wait(EventGeneration needed_gen);
	// create an event that won't trigger until all input events have
	Event merge_events(const std::set<Event> &wait_for);
	// Trigger the event
	void trigger(void);
	// Check to see if the lock is active, if not activate it (return true), otherwise false
	bool activate(void);	
	// Register a dependent event, return true if event had not been triggered and was registered
	bool register_dependent(Triggerable *target, EventGeneration needed_gen);
	// Return an event for this EventImplementation
	Event get_event();
    private: 
	bool in_use;
	int sources;
	EventIndex index;
	EventGeneration generation;
	pthread_mutex_t mutex;
	pthread_cond_t wait_cond;
	std::vector<Triggerable*> triggerables;
    }; 

    bool Event::has_triggered(void) const
    {
	if (!id) return true;
	EventImpl *e = Runtime::get_runtime()->get_event_impl(*this);
	return e->has_triggered(EventImpl::get_gen(id));
    }

    void Event::wait(void) const
    {
	if (!id) return;
	EventImpl *e = Runtime::get_runtime()->get_event_impl(*this);
	e->wait(EventImpl::get_gen(id));
    }

    Event Event::merge_events(const std::set<Event>& wait_for)
    {
	EventImpl *e = Runtime::get_runtime()->get_free_event();	
	return e->merge_events(wait_for);
    }

    bool EventImpl::has_triggered(EventGeneration needed_gen)
    {
	bool result = false;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	result = (needed_gen < generation);
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    void EventImpl::wait(EventGeneration needed_gen)
    {
	// First check to see if the event has triggered
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));	
	// Wait until the generation indicates that the event has occurred
	while (needed_gen >= generation) 
	{
		PTHREAD_SAFE_CALL(pthread_cond_wait(&wait_cond,&mutex));
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    Event EventImpl::merge_events(const std::set<Event> &wait_for)
    {
	// We need the lock here so that events we've already registered
	// can't trigger this event before sources is set
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	sources = 0;
	for (std::set<Event>::const_iterator it = wait_for.begin();
		it != wait_for.end(); it++)
	{
		EventImpl *src_impl = Runtime::get_runtime()->get_event_impl(*it);			
		if (src_impl->register_dependent(this,EventImpl::get_gen((*it).id)))
			sources++;
	}	
	Event ret;
	ret.id = make_id(index,generation);
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return ret;
    } 

    void EventImpl::trigger(void)
    {
	// Update the generation
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if (sources == 0)
	{
		generation++;
		in_use = false;
		// Trigger any dependent events
		while (!triggerables.empty())
		{
			triggerables.back()->trigger();
			triggerables.pop_back();
		}
		PTHREAD_SAFE_CALL(pthread_cond_broadcast(&wait_cond));
	}
	else
	{
		sources--;
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));	
    }

    bool EventImpl::activate(void)
    {
	bool result = false;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if (!in_use)
	{
		in_use = true;
		result = true;
		sources = 0;
	}	
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    bool EventImpl::register_dependent(Triggerable *target, EventGeneration gen)
    {
	bool result = false;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	// Make sure they're asking for the right generation, otherwise it's already triggered
	if (gen >= generation)
	{
		result = true;
		// Enqueue it
		triggerables.push_back(target);	
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));	
	return result;
    }

    Event EventImpl::get_event() 
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	Event result;
	result.id = make_id(index,generation);
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    ////////////////////////////////////////////////////////
    // Lock 
    ////////////////////////////////////////////////////////

    class LockImpl : public Triggerable {
    public:
	LockImpl(bool activate = false) {
		active = activate;
		taken = false;
		mode = 0;
		holders = 0;
		PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
	};	

	Event lock(unsigned mode, bool exclusive);
	void unlock(Event wait_on);
	void trigger(void);

	bool activate(void);
	void deactivate(void);
    private:
	Event register_request(unsigned m, bool exc);
	void perform_unlock();
    private:
	class LockRecord {
	public:
		unsigned mode;
		bool exclusive;
		Event event;
		bool handled;
	};
    private:
	bool active;
	bool taken;
	bool exclusive;
	unsigned mode;
	unsigned holders;
	std::list<LockRecord> requests;
	pthread_mutex_t mutex;
    };

    Event Lock::lock(unsigned mode, bool exclusive)
    {
	LockImpl *l = Runtime::get_runtime()->get_lock_impl(*this);
	return l->lock(mode,exclusive);
    }

    bool Lock::exists(void) const
    {
	return (id != 0);
    }

    void Lock::unlock(Event wait_on)
    {
	LockImpl *l = Runtime::get_runtime()->get_lock_impl(*this);
	l->unlock(wait_on);
    }

    Event LockImpl::lock(unsigned m, bool exc)
    {
	Event result = Event::NO_EVENT;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if (taken)
	{
		if (exclusive)
		{
			result = register_request(m,exc);
		}
		else
		{
			if (mode == m)
			{
				// Not exclusive and modes are equal
				// Can still acquire the lock
				holders++;
			}
			else
			{
				result = register_request(m,exc);	
			}
		}
	}
	else
	{
		// Nobody has the lock, grab it
		taken = true;
		exclusive = exc;
		mode = m;
		holders = 1;
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    // Always called while holding the lock
    Event LockImpl::register_request(unsigned m, bool exc)
    {
	EventImpl *e = Runtime::get_runtime()->get_free_event();
	LockRecord req;
	req.mode = m;
	req.exclusive = exc;
	req.event = e->get_event();
	req.handled = false;
	// Add this to the list of requests
	requests.push_back(req);
	return req.event;
    }

    void LockImpl::unlock(Event wait_on)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if (wait_on.exists())
	{
		// Register this lock to be unlocked when the even triggers	
		EventImpl *e = Runtime::get_runtime()->get_event_impl(wait_on);
		e->register_dependent(this,EventImpl::get_gen(wait_on.id));		
	}
	else
	{
		// No need to wait to perform the unlock
		perform_unlock();		
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    void LockImpl::trigger(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	// trigger the unlock operation now that the event has fired
	perform_unlock();
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    // Always called while holding the lock's mutex
    void LockImpl::perform_unlock(void)
    {
	holders--;	
	// If the holders are zero, get the next request out of the queue and trigger it
	if ((holders==0) && (!requests.empty()))
	{
		LockRecord req = requests.front();
		requests.pop_front();		
		// Get a request that hasn't already been handled
		while (req.handled && !requests.empty())
		{
			req = requests.front();
			requests.pop_front();
		}
		// If we emptied the queue with no unhandled requests, return
		if (req.handled)
			return;
		// Set the mode and exclusivity
		exclusive = req.exclusive;
		mode = req.mode;
		holders = 1;
		// Trigger the event
		Runtime::get_runtime()->get_event_impl(req.event)->trigger();
		// If this isn't an exclusive mode, see if there are any other
		// requests with the same mode that aren't exclusive that we can handle
		if (!exclusive)
		{
			for (std::list<LockRecord>::iterator it = requests.begin();
				it != requests.end(); it++)
			{
				if ((it->mode == mode) && (!it->exclusive) && (!it->handled))
				{
					it->handled = true;
					Runtime::get_runtime()->get_event_impl(it->event)->trigger();
					holders++;
				}
			}	
		}
	}
    }

    bool LockImpl::activate(void)
    {
	bool result = false;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if (!active)
	{
		active = true;
		result = true;
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    void LockImpl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	active = false;	
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    ////////////////////////////////////////////////////////
    // Processor 
    ////////////////////////////////////////////////////////

    class ProcessorImpl : public Triggerable {
    public:
	ProcessorImpl(Processor::TaskIDTable table, Processor p) 
		: scheduler(NULL), task_table(table), proc(p)
	{
		PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
		PTHREAD_SAFE_CALL(pthread_cond_init(&wait_cond,NULL));
	};
    public:
	Event spawn(Processor::TaskFuncID func_id, const void * args,
				size_t arglen, Event wait_on);
	void run(void);
	void register_scheduler(void (*scheduler)(Processor));
	void trigger(void);
	static void* start(void *proc);
    private:
	class TaskDesc {
	public:
		Processor::TaskFuncID func_id;
		void * args;
		size_t arglen;
		Event wait;
		EventImpl *complete;
	};
    private:
	void (*scheduler)(Processor);
	Processor::TaskIDTable task_table;
	Processor proc;
	std::list<TaskDesc> ready_queue;
	std::list<TaskDesc> waiting_queue;
	pthread_mutex_t mutex;
	pthread_cond_t wait_cond;
    };

    Event Processor::spawn(Processor::TaskFuncID func_id, const void * args,
				size_t arglen, Event wait_on) const
    {
	ProcessorImpl *p = Runtime::get_runtime()->get_processor_impl(*this);
	return p->spawn(func_id, args, arglen, wait_on);
    }

    bool Processor::exists(void) const
    {
	return (id != 0);
    }

    void Processor::register_scheduler(void (*scheduler)(Processor))
    {
	ProcessorImpl *p = Runtime::get_runtime()->get_processor_impl(*this);
	return p->register_scheduler(scheduler);
    }

    Event ProcessorImpl::spawn(Processor::TaskFuncID func_id, const void * args,
				size_t arglen, Event wait_on)
    {
	TaskDesc task;
	task.func_id = func_id;
	task.args = malloc(arglen);
	memcpy(task.args,args,arglen);
	task.arglen = arglen;
	task.wait = wait_on;
	task.complete = Runtime::get_runtime()->get_free_event();

	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if (wait_on.exists() && !wait_on.has_triggered())
	{
		// Put it on the waiting queue	
		waiting_queue.push_back(task);
		// Try registering this processor with the event
		EventImpl *wait_impl = Runtime::get_runtime()->get_event_impl(wait_on);
		if (!wait_impl->register_dependent(this, EventImpl::get_gen(wait_on.id)))
		{
			// If it wasn't registered, then the event triggered
			// Notify the processor thread in case it is waiting
			PTHREAD_SAFE_CALL(pthread_cond_signal(&wait_cond));
		}	
	}
	else
	{
		// Put it on the ready queue
		ready_queue.push_back(task);
		// Signal the thread there is a task to run in case it is waiting
		PTHREAD_SAFE_CALL(pthread_cond_signal(&wait_cond));
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return task.complete->get_event();
    }

    void ProcessorImpl::run(void)
    {
	// Processors run forever
	while (true)
	{
		// check to see if there are tasks to run
		PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
		// Check to see how many tasks there are
		// If there are too few, invoke the scheduler
		if ((scheduler!=NULL) &&(ready_queue.size()+waiting_queue.size()) < MIN_SCHED_TASKS)
		{
			PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
			scheduler(proc);
			PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
		}
		if (ready_queue.empty())
		{	
			// Look through the waiting queue, to see if any events
			// have been woken up	
			for (std::list<TaskDesc>::iterator it = waiting_queue.begin();
				it != waiting_queue.end(); it++)
			{
				if (it->wait.has_triggered())
				{
					ready_queue.push_back(*it);
					waiting_queue.erase(it);
					break;
				}	
			}	
			// Wait until someone tells us there is work to do
			if (ready_queue.empty())
			{
				PTHREAD_SAFE_CALL(pthread_cond_wait(&wait_cond,&mutex));
			}
			PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
		}
		else
		{
			// Pop a task off the queue and run it
			TaskDesc task = ready_queue.front();
			ready_queue.pop_front();
			PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));	
#ifdef DEBUG_LOW_LEVEL
			assert(task_table.find(task.func_id) != task_table.end());
#endif
			Processor::TaskFuncPtr func = task_table[task.func_id];	
			func(task.args, task.arglen, proc);
			// Trigger the event indicating that the task has been run
			task.complete->trigger();
			// Clean up the mess
			free(task.args);
		}
	}
    }

    void ProcessorImpl::register_scheduler(void (*sched)(Processor))
    {
	scheduler = sched;	
    }

    void ProcessorImpl::trigger(void)
    {
	// We're not sure which task is ready, but at least one of them is
	// so wake up the processor thread if it is waiting
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	PTHREAD_SAFE_CALL(pthread_cond_signal(&wait_cond));
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    // The static method used to start the processor running
    void* ProcessorImpl::start(void *p)
    {
	ProcessorImpl *proc = (ProcessorImpl*)p;
	// Set the thread local variable processor id
	local_proc_id = proc->proc.id;
	proc->run();
	pthread_exit(NULL);	
    }

    ////////////////////////////////////////////////////////
    // Memory 
    ////////////////////////////////////////////////////////

    class MemoryImpl {
    public:
	MemoryImpl(size_t max) 
		: max_size(max), remaining(max)
	{
		PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
	}
    public:
	size_t remaining_bytes(void);
	void* allocate_space(size_t size);
	void free_space(void *ptr, size_t size);
    private:
	const size_t max_size;
	size_t remaining;
	pthread_mutex_t mutex;
    };

    bool Memory::exists(void) const
    {
	MemoryImpl* m = Runtime::get_runtime()->get_memory_impl(*this);
	return (m!=NULL);
    }

    size_t MemoryImpl::remaining_bytes(void) 
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	size_t result = remaining;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    void* MemoryImpl::allocate_space(size_t size)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	void *ptr = NULL;
	if (size < remaining)
	{
		remaining -= size;
		ptr = malloc(size);
#ifdef DEBUG_LOW_LEVEL
		assert(ptr != NULL);
#endif
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return ptr;
    }

    void MemoryImpl::free_space(void *ptr, size_t size)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
#ifdef DEBUG_LOW_LEVEL
	assert(ptr != NULL);
#endif
	remaining += size;
	free(ptr);
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    
    ////////////////////////////////////////////////////////
    // RegionMetaDataUntyped 
    ////////////////////////////////////////////////////////

    class RegionMetaDataImpl {
    public:
	RegionMetaDataImpl(int idx) {
		PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
		active = false;
		index = idx;
	}
    public:
	bool activate(Memory m, size_t num_elmts, size_t elmt_size);
	void deactivate(void);	
	RegionMetaDataUntyped get_metadata(void);

	RegionAllocatorUntyped create_allocator(Memory m);
	RegionInstanceUntyped  create_instance(Memory m);

	void destroy_allocator(RegionAllocatorUntyped a);
	void destroy_instance(RegionInstanceUntyped i);

	Lock create_lock(void);
	void destroy_lock(Lock l);

	RegionAllocatorUntyped get_master_allocator(void);
	RegionInstanceUntyped get_master_instance(void);
	
	void set_master_allocator(RegionAllocatorUntyped a);
	void set_master_instance(RegionInstanceUntyped i);	
    private:
	RegionAllocatorUntyped master_allocator;
	RegionInstanceUntyped  master_instance;
	std::set<RegionAllocatorUntyped> allocators;
	std::set<RegionInstanceUntyped> instances;
	std::set<Lock> locks;
	pthread_mutex_t mutex;
	bool active;
	int index;
	size_t num_elmts;
	size_t elmt_size;
    };

    RegionMetaDataUntyped RegionMetaDataUntyped::create_region_untyped(Memory m, size_t num_elmts, size_t elmt_size)
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_free_metadata(m, num_elmts, elmt_size);	
	return r->get_metadata();
    }

    RegionAllocatorUntyped RegionMetaDataUntyped::create_allocator_untyped(Memory m)
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	return r->create_allocator(m);
    }

    RegionInstanceUntyped RegionMetaDataUntyped::create_instance_untyped(Memory m)
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	return r->create_instance(m);
    }

    void RegionMetaDataUntyped::destroy_region_untyped(void)
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	r->deactivate();
    }

    void RegionMetaDataUntyped::destroy_allocator_untyped(RegionAllocatorUntyped a)
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	r->destroy_allocator(a);
    }

    void RegionMetaDataUntyped::destroy_instance_untyped(RegionInstanceUntyped i)
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	r->destroy_instance(i);
    }

    Lock RegionMetaDataUntyped::create_lock(void)
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	return r->create_lock();
    }

    void RegionMetaDataUntyped::destroy_lock(Lock l)
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	r->destroy_lock(l);
    }

    RegionAllocatorUntyped RegionMetaDataUntyped::get_master_allocator_untyped(void)
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	return r->get_master_allocator();
    }

    RegionInstanceUntyped RegionMetaDataUntyped::get_master_instance_untyped(void)
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	return r->get_master_instance();
    }

    void RegionMetaDataUntyped::set_master_allocator_untyped(RegionAllocatorUntyped a)
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	r->set_master_allocator(a);
    }

    void RegionMetaDataUntyped::set_master_instance_untyped(RegionInstanceUntyped i)
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	r->set_master_instance(i);
    }

    bool RegionMetaDataImpl::activate(Memory m, size_t num, size_t size)
    {
	bool result = false;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if (!active)
	{ 
		active = true;
		result = true;
		num_elmts = num;
		elmt_size = size;
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    void RegionMetaDataImpl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	for (std::set<RegionAllocatorUntyped>::iterator it = allocators.begin();
		it != allocators.end(); it++)
	{

	}	
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    RegionMetaDataUntyped RegionMetaDataImpl::get_metadata(void)
    {

    }

    RegionAllocatorUntyped RegionMetaDataImpl::create_allocator(Memory m)
    {

    }

    RegionInstanceUntyped RegionMetaDataImpl::create_instance(Memory m)
    {

    }

    void RegionMetaDataImpl::destroy_allocator(RegionAllocatorUntyped a)
    {

    }

    void RegionMetaDataImpl::destroy_instance(RegionInstanceUntyped i)
    {

    }

    Lock RegionMetaDataImpl::create_lock(void)
    {

    }

    void RegionMetaDataImpl::destroy_lock(Lock l)
    {

    }

    RegionAllocatorUntyped RegionMetaDataImpl::get_master_allocator(void)
    {

    }

    RegionInstanceUntyped RegionMetaDataImpl::get_master_instance(void)
    {

    }

    void RegionMetaDataImpl::set_master_allocator(RegionAllocatorUntyped a)
    {

    }

    void RegionMetaDataImpl::set_master_instance(RegionInstanceUntyped i)	
    {

    }

    ////////////////////////////////////////////////////////
    // Machine 
    ////////////////////////////////////////////////////////

    Machine::Machine(int *argc, char ***argv,
			const Processor::TaskIDTable &task_table)
    {
	// Create the runtime and initialize with this machine
	Runtime::runtime = new Runtime(this);
	
	// Fill in the tables
	for (int id=0; id<NUM_PROCS; id++)
	{
		Processor p;
		p.id = id;
		procs.insert(p);
		ProcessorImpl *impl = new ProcessorImpl(task_table, p);
		Runtime::runtime->processors.push_back(impl);
	}	
	{
		Memory global;
		global.id = 0;
		memories.insert(global);
		MemoryImpl *impl = new MemoryImpl(MAX_GLOBAL_MEM);
		Runtime::runtime->memories.push_back(impl);
	}
	for (int id=1; id<=NUM_PROCS; id++)
	{
		Memory m;
		m.id = id;
		memories.insert(m);
		MemoryImpl *impl = new MemoryImpl(MAX_LOCAL_MEM);
		Runtime::runtime->memories.push_back(impl);
	}
	// All memories are visible from each processor
	for (int id=0; id<NUM_PROCS; id++)
	{
		Processor p;
		p.id = id;
		visible_memories_from_procs.insert(std::pair<Processor,std::set<Memory> >(p,memories));
	}	
	// All memories are visible from all memories, all processors are visible from all memories
	for (int id=0; id<=NUM_PROCS; id++)
	{
		Memory m;
		m.id = id;
		visible_memories_from_memory.insert(std::pair<Memory,std::set<Memory> >(m,memories));
		visible_procs_from_memory.insert(std::pair<Memory,std::set<Processor> >(m,procs));
	}

	// Now start the threads for each of the processors
	for (int id=0; id<NUM_PROCS; id++)
	{
		ProcessorImpl *impl = Runtime::runtime->processors[id];
		pthread_t thread;
		PTHREAD_SAFE_CALL(pthread_create(&thread, NULL, ProcessorImpl::start, (void*)impl));
	}
    }

    const std::set<Memory>& Machine::get_visible_memories(const Processor p) 
    {
#ifdef DEBUG_LOW_LEVEL
	assert(visible_memories_from_procs.find(p) != visible_memories_from_procs.end());
#endif
	return visible_memories_from_procs[p];	
    }

    const std::set<Memory>& Machine::get_visible_memories(const Memory m)
    {
#ifdef DEBUG_LOW_LEVEL
	assert(visible_memories_from_memory.find(m) != visible_memories_from_memory.end());
#endif
	return visible_memories_from_memory[m];
    }

    const std::set<Processor>& Machine::get_shared_processors(const Memory m)
    {
#ifdef DEBUG_LOW_LEVEL
	assert(visible_procs_from_memory.find(m) != visible_procs_from_memory.end());
#endif
	return visible_procs_from_memory[m];
    }

    Processor Machine::get_local_processor() const
    {
	Processor p;
	p.id = local_proc_id;
	return p;
    }

    Machine::ProcessorKind Machine::get_processor_kind(Processor p) const
    {
	return LOC_PROC;
    }

    size_t Machine::get_memory_size(const Memory m) const
    {
	if (m.id == 0)
		return MAX_GLOBAL_MEM;
	else
		return MAX_LOCAL_MEM;
    }

    Machine* Machine::get_machine(void)
    {
	return Runtime::get_runtime()->machine;
    }
    


    ////////////////////////////////////////////////////////
    // Runtime 
    ////////////////////////////////////////////////////////

    Runtime::Runtime(Machine *m)
	: machine(m)
    {
	for (unsigned i=0; i<BASE_EVENTS; i++)
		events.push_back(new EventImpl(i));

	for (unsigned i=0; i<BASE_LOCKS; i++)
		locks.push_back(new LockImpl());

	PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
    }

    EventImpl* Runtime::get_event_impl(Event e)
    {
	EventImpl::EventIndex i = EventImpl::get_index(e.id);
#ifdef DEBUG_LOW_LEVEL
	assert(i < events.size());
#endif
	return events[i];
    }

    LockImpl* Runtime::get_lock_impl(Lock l)
    {
#ifdef DEBUG_LOW_LEVEL
	assert(l.id < locks.size());
#endif
	return locks[l.id];
    }

    MemoryImpl* Runtime::get_memory_impl(Memory m)
    {
	if (m.id < memories.size())
		return memories[m.id];
	else
		return NULL;
    }

    ProcessorImpl* Runtime::get_processor_impl(Processor p)
    {
#ifdef DEBUG_LOW_LEVEL
	assert(p.id < processors.size());
#endif
	return processors[p.id];
    }

    RegionMetaDataImpl* Runtime::get_metadata_impl(RegionMetaDataUntyped m)
    {
#ifdef DEBUG_LOW_LEVEL
	assert(m.id < metadatas.size());
#endif
	return metadatas[m.id];
    }

    RegionAllocatorImpl* Runtime::get_allocator_impl(RegionAllocatorUntyped a)
    {
#ifdef DEBUG_LOW_LEVEL
	assert(a.id < allocators.size());
#endif
	return allocators[a.id];
    }

    RegionInstanceImpl* Runtime::get_instance_impl(RegionInstanceUntyped i)
    {
#ifdef DEBUG_LOW_LEVEL
	assert(i.id < instances.size());
#endif
	return instances[i.id];
    }

    EventImpl* Runtime::get_free_event()
    {
	// Iterate over the events looking for a free one
	for (unsigned i=1; i<events.size(); i++)
	{
		if (events[i]->activate())
		{
			return events[i];
		}
	}
	// Otherwise there are no free events so make a new one
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	unsigned index = events.size();
	events.push_back(new EventImpl(index, true));
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return events[index];
    }

    LockImpl* Runtime::get_free_lock()
    {
	// Iterate over the locks looking for a free one
	for (unsigned int i=1; i<locks.size(); i++)
	{
		if (locks[i]->activate())
		{
			return locks[i];
		}
	}
	// Otherwise there are no free locks so make a new one
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	unsigned index = locks.size();
	locks.push_back(new LockImpl(true));
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return locks[index];
    }

    RegionMetaDataImpl* Runtime::get_free_metadata(Memory m, size_t num_elmts, size_t elmt_size)
    {
	for (unsigned int i=0; i<metadatas.size(); i++)
	{
		if (metadatas[i]->activate(m,num_elmts,elmt_size))
		{
			return metadatas[i];
		}
	}
	// Otherwise there are no free metadata so make a new one
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	unsigned int index = metadatas.size();
	metadatas.push_back(new RegionMetaDataImpl(index));
	metadatas[index]->activate(m,num_elmts,elmt_size);
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return metadatas[index];
    }
  };
};
