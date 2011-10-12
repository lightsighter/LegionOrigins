
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
#define BASE_METAS	64
#define BASE_ALLOCATORS	64
#define BASE_INSTANCES	64

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

#ifdef DEBUG_PRINT
#define DPRINT1(str,arg)						\
	{								\
		PTHREAD_SAFE_CALL(pthread_mutex_lock(&debug_mutex));	\
		fprintf(stderr,str,arg);				\
		fflush(stderr);						\
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(&debug_mutex));	\
	}

#define DPRINT2(str,arg1,arg2)						\
	{								\
		PTHREAD_SAFE_CALL(pthread_mutex_lock(&debug_mutex));	\
		fprintf(stderr,str,arg1,arg2);				\
		fflush(stderr);						\
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(&debug_mutex));	\
	}

#define DPRINT3(str,arg1,arg2,arg3)					\
	{								\
		PTHREAD_SAFE_CALL(pthread_mutex_lock(&debug_mutex));	\
		fprintf(stderr,str,arg1,arg2,arg3);			\
		fflush(stderr);						\
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(&debug_mutex));	\
	}

#define DPRINT4(str,arg1,arg2,arg3,arg4)				\
	{								\
		PTHREAD_SAFE_CALL(pthread_mutex_lock(&debug_mutex));	\
		fprintf(stderr,str,arg1,arg2,arg3,arg4);		\
		fflush(stderr);						\
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(&debug_mutex));	\
	}

// Declration for the debug mutex
pthread_mutex_t debug_mutex;
#endif // DEBUG_PRINT

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
      RegionAllocatorImpl* get_free_allocator(unsigned start, unsigned end);
      RegionInstanceImpl*  get_free_instance(Memory m, size_t num_elmts, size_t elmt_size);
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
      pthread_rwlock_t event_lock;
      pthread_rwlock_t lock_lock;
      pthread_rwlock_t metadata_lock;
      pthread_rwlock_t allocator_lock;
      pthread_rwlock_t instance_lock;
    };

    /* static */
    Runtime *Runtime::runtime = NULL;

    __thread unsigned local_proc_id;

    // Any object which can be triggered should be able to triggered
    // This will include Events and Locks
    class Triggerable {
    public:
	virtual void trigger(void) = 0;
	// make the warnings go away
	virtual ~Triggerable() { }
    };

    ////////////////////////////////////////////////////////
    // Event Impl (up here since we need it in Processor Impl) 
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
	EventImpl(EventIndex idx, bool activate=false) 
		: index(idx)
	{
	  in_use = activate;
	  generation = 0;
	  sources = 0;
	  PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
	  PTHREAD_SAFE_CALL(pthread_cond_init(&wait_cond,NULL));
	  if (in_use)
	  {
	    // Always initialize the current event to hand out to
	    // generation + 1, so the event will have triggered
	    // when the event matches the generation
	    current.id = make_id(index,generation+1);
	    sources = 1;
#ifdef DEBUG_LOW_LEVEL
	    assert(current.exists());
#endif
          }
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
	const EventIndex index;
	EventGeneration generation;
	// The version of the event to hand out (i.e. with generation+1)
	// so we can detect when the event has triggered with testing
	// generational equality
	Event current; 
	pthread_mutex_t mutex;
	pthread_cond_t wait_cond;
	std::vector<Triggerable*> triggerables;
    }; 

    ////////////////////////////////////////////////////////
    // Processor Impl (up here since we need it in Event) 
    ////////////////////////////////////////////////////////

    class ProcessorImpl : public Triggerable {
    public:
	ProcessorImpl(Processor::TaskIDTable table, Processor p) 
		: scheduler(NULL), task_table(table), proc(p)
	{
		PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
		PTHREAD_SAFE_CALL(pthread_cond_init(&wait_cond,NULL));
		shutdown = false;
		shutdown_trigger = NULL;
	};
    public:
	Event spawn(Processor::TaskFuncID func_id, const void * args,
				size_t arglen, Event wait_on);
	void run(void);
	void register_scheduler(void (*scheduler)(Processor));
	void trigger(void);
	static void* start(void *proc);
	void preempt(EventImpl *event, EventImpl::EventGeneration needed);
    private:
	void execute_task(bool permit_shutdown);
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
	// Used for detecting the shutdown condition
	bool shutdown;
	EventImpl *shutdown_trigger;
    };

    
    ////////////////////////////////////////////////////////
    // Events 
    ////////////////////////////////////////////////////////

    /* static */ const Event Event::NO_EVENT = Event();

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
	result = (needed_gen <= generation);
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    void EventImpl::wait(EventGeneration needed_gen)
    {
#if 0 // This version blocks the processor
	// First check to see if the event has triggered
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));	
	// Wait until the generation indicates that the event has occurred
	while (needed_gen >= generation) 
	{
		PTHREAD_SAFE_CALL(pthread_cond_wait(&wait_cond,&mutex));
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
#else // This version doesn't block the processor
	// Try preempting the process
	Processor local = { local_proc_id };
	ProcessorImpl *impl = Runtime::get_runtime()->get_processor_impl(local);
	// This call will only return once the event has triggered
	impl->preempt(this,needed_gen);
#endif
    }

    Event EventImpl::merge_events(const std::set<Event> &wait_for)
    {
	// We need the lock here so that events we've already registered
	// can't trigger this event before sources is set
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
#ifdef DEBUG_PRINT
	//DPRINT2("Mering events into event %u generation %u\n",index,generation);
#endif
	sources = 0;
	for (std::set<Event>::const_iterator it = wait_for.begin();
		it != wait_for.end(); it++)
	{
		if (!(*it).exists())
			continue;
		EventImpl *src_impl = Runtime::get_runtime()->get_event_impl(*it);			
		// Handle the special case where this event is an older generation
		// of the same event implementation.  In this case we know it
		// already triggered.
		if (src_impl == this)
			continue;
		if (src_impl->register_dependent(this,EventImpl::get_gen((*it).id)))
			sources++;
	}	
	Event ret = current;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return ret;
    } 

    void EventImpl::trigger(void)
    {
	// Update the generation
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
#ifdef DEBUG_LOW_LEVEL
	assert(sources > 0);
#endif
	sources--;
	if (sources == 0)
	{
#ifdef DEBUG_PRINT
		//DPRINT2("Event %u triggered for generation %u\n",index,generation);
#endif
		// Increment the generation so that nobody can register a triggerable
		// with this event, but keep event in_use so no one can use the event
		generation++;
#ifdef DEBUG_LOW_LEVEL
		assert(generation == EventImpl::get_gen(current.id));
#endif
		// Can't be holding the lock when triggering other triggerables
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
		// Trigger any dependent events
		while (!triggerables.empty())
		{
			triggerables.back()->trigger();
			triggerables.pop_back();
		}
		// Reacquire the lock and mark that in_use is false
		PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
		in_use = false;
		// Wake up any waiters
		PTHREAD_SAFE_CALL(pthread_cond_broadcast(&wait_cond));
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
		sources = 1;
		// Set generation to generation+1, see 
		// comment in constructor
		current.id = make_id(index,generation+1);
#ifdef DEBUG_LOW_LEVEL
		assert(current.exists());
#endif
	}	
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    bool EventImpl::register_dependent(Triggerable *target, EventGeneration gen)
    {
	bool result = false;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	// Make sure they're asking for the right generation, otherwise it's already triggered
	if (gen > generation)
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
	Event result = current;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    ////////////////////////////////////////////////////////
    // Lock 
    ////////////////////////////////////////////////////////

    class LockImpl : public Triggerable {
    public:
	LockImpl(int idx, bool activate = false) : index(idx) {
		active = activate;
		taken = false;
		mode = 0;
		holders = 0;
		waiters = false;
		PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
	};	

	Event lock(unsigned mode, bool exclusive);
	void unlock(Event wait_on);
	void trigger(void);

	bool activate(void);
	void deactivate(void);
	Lock get_lock(void) const;
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
	const int index;
	bool active;
	bool taken;
	bool exclusive;
	bool waiters;
	unsigned mode;
	unsigned holders;
	std::list<LockRecord> requests;
	pthread_mutex_t mutex;
    };

    Event Lock::lock(unsigned mode, bool exclusive, Event wait_on) const
    {
	LockImpl *l = Runtime::get_runtime()->get_lock_impl(*this);
	return l->lock(mode,exclusive);
    }

    bool Lock::exists(void) const
    {
	return (id != 0);
    }

    void Lock::unlock(Event wait_on) const
    {
	LockImpl *l = Runtime::get_runtime()->get_lock_impl(*this);
	l->unlock(wait_on);
    }

    Lock Lock::create_lock(void)
    {
	return Runtime::get_runtime()->get_free_lock()->get_lock();
    }

    void Lock::destroy_lock(void)
    {
	LockImpl *l = Runtime::get_runtime()->get_lock_impl(*this);
	l->deactivate();
    }

    Event LockImpl::lock(unsigned m, bool exc)
    {
	Event result = Event::NO_EVENT;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if (taken)
	{
		// If either is exclusive we have to register the request
		if (exclusive || exc)
		{
			result = register_request(m,exc);
		}
		else
		{
			if ((mode == m) && !waiters)
			{
				// Not exclusive and modes are equal
				// and there are no waiters
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
#ifdef DEBUG_PRINT
		DPRINT3("Granting lock %d in mode %d with exclusive %d\n",index,mode,exclusive);
#endif
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

	// Finally set waiters to true
	// as there are now threads waiting
	waiters = true;
	
	return req.event;
    }

    void LockImpl::unlock(Event wait_on)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if (wait_on.exists())
	{
		// Register this lock to be unlocked when the even triggers	
		EventImpl *e = Runtime::get_runtime()->get_event_impl(wait_on);
		if (!(e->register_dependent(this,EventImpl::get_gen(wait_on.id))))
		{
			// The event didn't register which means it already triggered
			// so go ahead and perform the unlock operation
			perform_unlock();
		}	
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
	if ((holders==0))
	{
#ifdef DEBUG_PRINT
		DPRINT1("Unlocking lock %d\n",index);
#endif
		// Check to see if there are any waiters
		if (requests.empty())
		{
			waiters= false;
			taken = false;
			return;
		}
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
		{
			waiters = false;
			taken = false;
			return;
		}
		// Set the mode and exclusivity
		exclusive = req.exclusive;
		mode = req.mode;
		holders = 1;
#ifdef DEBUG_PRINT
		DPRINT3("Issuing lock %d in mode %d with exclusivity %d\n",index,mode,exclusive);
#endif
		// Trigger the event
		Runtime::get_runtime()->get_event_impl(req.event)->trigger();
		// If this isn't an exclusive mode, see if there are any other
		// requests with the same mode that aren't exclusive that we can handle
		if (!exclusive)
		{
			waiters = false;
			for (std::list<LockRecord>::iterator it = requests.begin();
				it != requests.end(); it++)
			{
				if ((it->mode == mode) && (!it->exclusive) && (!it->handled))
				{
					it->handled = true;
					Runtime::get_runtime()->get_event_impl(it->event)->trigger();
					holders++;
				}
				else
				{
					// There is at least one thread still waiting
					waiters = true;
				}
			}	
		}
		else
		{
			waiters = (requests.size()>0);
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
		waiters = false;
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

    Lock LockImpl::get_lock(void) const
    {
	Lock l;
	l.id = index;
	return l;
    }

    ////////////////////////////////////////////////////////
    // Processor 
    ////////////////////////////////////////////////////////

    // Processor Impl at top due to use in event
    
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

#if 0
    void Processor::register_scheduler(void (*scheduler)(Processor))
    {
	ProcessorImpl *p = Runtime::get_runtime()->get_processor_impl(*this);
	return p->register_scheduler(scheduler);
    }
#endif

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
	Event result = task.complete->get_event();

	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if (wait_on.exists())
	{
		// Try registering this processor with the event
		EventImpl *wait_impl = Runtime::get_runtime()->get_event_impl(wait_on);
		if (!wait_impl->register_dependent(this, EventImpl::get_gen(wait_on.id)))
		{
#ifdef DEBUG_PRINT
			DPRINT2("Registering task %d on processor %d ready queue\n",func_id,proc.id);
#endif
			// Failed to register which means it is ready to execute
			ready_queue.push_back(task);
			// If it wasn't registered, then the event triggered
			// Notify the processor thread in case it is waiting
			PTHREAD_SAFE_CALL(pthread_cond_signal(&wait_cond));
		}	
		else
		{
#ifdef DEBUG_PRINT
			DPRINT2("Registering task %d on processor %d waiting queue\n",func_id,proc.id);
#endif
			// Successfully registered, put the task on the waiting queue
			waiting_queue.push_back(task);
		}
	}
	else
	{
#ifdef DEBUG_PRINT
		DPRINT2("Putting task %d on processor %d ready queue\n",func_id,proc.id);
#endif
		// Put it on the ready queue
		ready_queue.push_back(task);
		// Signal the thread there is a task to run in case it is waiting
		PTHREAD_SAFE_CALL(pthread_cond_signal(&wait_cond));
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    void ProcessorImpl::run(void)
    {
	// Processors run forever and permit shutdowns
	while (true)
	{
		// Make sure we're holding the lock
		PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
		// This task will perform the unlock
		execute_task(true);
	}
    }

    void ProcessorImpl::preempt(EventImpl *event, EventImpl::EventGeneration needed)
    {
	// Try registering this processor with the event in case it goes to sleep
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if (!(event->register_dependent(this, needed)))
	{
		// The even triggered, release the lock and return
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
		return;
	}
	
	// Run until the event has been triggered
	while (!(event->has_triggered(needed)))
	{
		// Don't permit shutdowns since there is still a task waiting
		execute_task(false);
		// Relock the task for our next attempt
		PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	}

	// Unlock and return
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    // Must always be holding the lock when calling this task
    // This task will always unlock it
    void ProcessorImpl::execute_task(bool permit_shutdown)
    {
	// Check to see how many tasks there are
	// If there are too few, invoke the scheduler
	if ((scheduler!=NULL) &&(ready_queue.size()+waiting_queue.size()) < MIN_SCHED_TASKS)
	{
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
		scheduler(proc);
		// Return from the scheduler, so we can reevaluate status
		return;
	}
	if (ready_queue.empty())
	{	
		if (shutdown && permit_shutdown && waiting_queue.empty())
		{
			shutdown_trigger->trigger();
			pthread_exit(NULL);	
		}
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
		// Check for the shutdown function
		if (task.func_id == 0)
		{
			shutdown = true;
			shutdown_trigger = task.complete;
			// Continue going around until all tasks are run
			return;
		}
#ifdef DEBUG_LOW_LEVEL
		assert(task_table.find(task.func_id) != task_table.end());
#endif
		Processor::TaskFuncPtr func = task_table[task.func_id];	
		func(task.args, task.arglen, proc);
		// Trigger the event indicating that the task has been run
		task.complete->trigger();
		// Clean up the mess
		if (task.arglen > 0)
			free(task.args);
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
	// Will never return from this call
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
    // Region Allocator 
    ////////////////////////////////////////////////////////

    class RegionAllocatorImpl {
    public:
	RegionAllocatorImpl(int idx, unsigned s, unsigned e, bool activate = false) 
		: start(s), end(e), index(idx)
	{
		PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
		active = activate;
		if (active)
		{
			lock = Runtime::get_runtime()->get_free_lock();
		}
	}
    public:
	unsigned alloc_elmt(size_t num_elmts = 1);
	void free_elmt(unsigned ptr);	
	bool activate(unsigned s, unsigned e);
	void deactivate();
	RegionAllocatorUntyped get_allocator(void) const;
	Lock get_lock(void);
    private:
	unsigned start;
	unsigned end;	/*exclusive*/
	std::set<unsigned> free_list;
	pthread_mutex_t mutex;
	bool active;
	LockImpl *lock;
	const int index;
    }; 

    bool RegionAllocatorUntyped::exists(void) const
    {
	//RegionAllocatorImpl *allocator = Runtime::get_runtime()->get_allocator_impl(*this);
	//return (allocator != NULL);
	return (id != 0);
    } 

    static unsigned alloc_func(RegionAllocatorUntyped region, size_t num_elmts)
    {
	return Runtime::get_runtime()->get_allocator_impl(region)->alloc_elmt(num_elmts);
    }

    RegionAllocatorUntyped::AllocFuncPtr RegionAllocatorUntyped::alloc_fn_untyped(void) const
    {
      return alloc_func;
    }

    static void free_func(RegionAllocatorUntyped region, unsigned ptr)
    {
	Runtime::get_runtime()->get_allocator_impl(region)->free_elmt(ptr);
    }

    RegionAllocatorUntyped::FreeFuncPtr RegionAllocatorUntyped::free_fn_untyped(void) const
    {
      return free_func;
    }

    unsigned RegionAllocatorImpl::alloc_elmt(size_t num_elmts)
    {
	unsigned result;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if ((start+num_elmts) <= end)
	{
		result = start;
		start += num_elmts;
	}
	else if (num_elmts == 1)
	{
		if (!free_list.empty())
		{
			std::set<unsigned>::iterator it = free_list.begin();
			result = *it;
			free_list.erase(it);
		}	
		else
			assert(false);
	}
	else
	{
		assert(false);
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    void RegionAllocatorImpl::free_elmt(unsigned ptr)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	free_list.insert(ptr);	
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    bool RegionAllocatorImpl::activate(unsigned s, unsigned e)
    {
	bool result = false;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if (!active)
	{
		active = true;
		start = s;
		end = e;
		lock = Runtime::get_runtime()->get_free_lock();
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    void RegionAllocatorImpl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	active = false;
	free_list.clear();
	start = 0;
	end = 0;
	lock->deactivate();
	lock = NULL;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    RegionAllocatorUntyped RegionAllocatorImpl::get_allocator(void) const
    {
	RegionAllocatorUntyped allocator;
	allocator.id = index;
	return allocator;
    }

    Lock RegionAllocatorImpl::get_lock(void)
    {
	return lock->get_lock();
    }

    
    ////////////////////////////////////////////////////////
    // Region Instance 
    ////////////////////////////////////////////////////////

    class RegionInstanceImpl : public Triggerable { 
    public:
	RegionInstanceImpl(int idx, Memory m, size_t num, size_t elem_size, bool activate = false)
		: elmt_size(elem_size), num_elmts(num), index(idx)
	{
		PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
		active = activate;
		if (active)
		{
			memory = m;
			// Use the memory to allocate the space, fail if there is none
			MemoryImpl *mem = Runtime::get_runtime()->get_memory_impl(m);
			base_ptr = (char*)mem->allocate_space(num_elmts*elem_size);	
#ifdef DEBUG_LOW_LEVEL
			assert(base_ptr != NULL);
#endif
			lock = Runtime::get_runtime()->get_free_lock();
		}
	}
    public:
	const void* read(unsigned ptr);
	void write(unsigned ptr, const void* newval);	
	bool activate(Memory m, size_t num_elmts, size_t elem_size);
	void deactivate(void);
	Event copy_to(RegionInstanceUntyped target, Event wait_on);
	RegionInstanceUntyped get_instance(void) const;
	void trigger(void);
	Lock get_lock(void);
    private:
	char *base_ptr;	
	size_t elmt_size;
	size_t num_elmts;
	Memory memory;
	pthread_mutex_t mutex;
	bool active;
	const int index;
	// Fields for the copy operation
	EventImpl *complete;
	LockImpl *lock;
	char *target_ptr;
    };

    bool RegionInstanceUntyped::exists(void) const
    {
	//RegionInstanceImpl *inst = Runtime::get_runtime()->get_instance_impl(*this);
	//return (inst != NULL);
	return (id != 0);
    }

    RegionInstanceAccessorUntyped<AccessorGeneric> RegionInstanceUntyped::get_accessor_untyped(void) const
    {
      RegionInstanceImpl *impl = Runtime::get_runtime()->get_instance_impl(*this);
      return RegionInstanceAccessorUntyped<AccessorGeneric>((void *)impl);
    }

    const void* read_func(RegionInstanceUntyped region, unsigned ptr)
    {
	return Runtime::get_runtime()->get_instance_impl(region)->read(ptr);
    }

    RegionInstanceUntyped::ReadFuncPtr RegionInstanceUntyped::read_fn_untyped(void)
    {
      return read_func;
    }

    void write_func(RegionInstanceUntyped region, unsigned ptr, const void* newval)
    {
	Runtime::get_runtime()->get_instance_impl(region)->write(ptr,newval);
    }

    RegionInstanceUntyped::WriteFuncPtr RegionInstanceUntyped::write_fn_untyped(void)
    {
      return write_func;
    }

    Event copy_to_func(RegionInstanceUntyped source, RegionInstanceUntyped target, Event wait_on)
    {
	return Runtime::get_runtime()->get_instance_impl(source)->copy_to(target, wait_on);
    }

    RegionInstanceUntyped::CopyFuncPtr RegionInstanceUntyped::copy_fn_untyped(void)
    {
      return copy_to_func;
    }

    Lock RegionInstanceUntyped::get_lock(void)
    {
	return Runtime::get_runtime()->get_instance_impl(*this)->get_lock();
    }

    const void* RegionInstanceImpl::read(unsigned ptr)
    {
	return ((void*)(base_ptr+(ptr*elmt_size)));
    }

    void RegionInstanceImpl::write(unsigned ptr, const void* newval)
    {
	memcpy((base_ptr+(ptr*elmt_size)),newval,elmt_size);
    }

    bool RegionInstanceImpl::activate(Memory m, size_t num, size_t elem_size)
    {
	bool result = false;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	if (!active)
	{
		active = true;
		result = true;
		memory = m;
		num_elmts = num;
		elmt_size = elem_size;
		MemoryImpl *mem = Runtime::get_runtime()->get_memory_impl(m);
		base_ptr = (char*)mem->allocate_space(num_elmts*elmt_size);
#ifdef DEBUG_LOW_LEVEL
		assert(base_ptr != NULL);
#endif
		lock = Runtime::get_runtime()->get_free_lock();
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    void RegionInstanceImpl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	active = false;
	MemoryImpl *mem = Runtime::get_runtime()->get_memory_impl(memory);
	mem->free_space(base_ptr,num_elmts*elmt_size);
	num_elmts = 0;
	elmt_size = 0;
	base_ptr = NULL;	
	lock->deactivate();
	lock = NULL;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    Event RegionInstanceImpl::copy_to(RegionInstanceUntyped target, Event wait_on)
    {
	RegionInstanceImpl *target_impl = Runtime::get_runtime()->get_instance_impl(target);
#ifdef DEBUG_LOW_LEVEL
	assert(target_impl->num_elmts == num_elmts);
	assert(target_impl->elmt_size == elmt_size);
#endif
	target_ptr = target_impl->base_ptr;
	// Check to see if the event exists
	if (wait_on.exists())
	{
		// TODO: Need to handle multiple outstanding copies 

		// Try registering this as a triggerable with the event	
		EventImpl *event_impl = Runtime::get_runtime()->get_event_impl(wait_on);
		PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
		if (event_impl->register_dependent(this,EventImpl::get_gen(wait_on.id)))
		{
			// Get a free event
			EventImpl *complete = Runtime::get_runtime()->get_free_event();
			PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
			return complete->get_event();
		}
		else
		{
			PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
			// The event occurred do the copy and return
			memcpy(target_ptr,base_ptr,num_elmts*elmt_size);
			return Event::NO_EVENT;
		}
	}
	else
	{
		// It doesn't exist, do the memcpy and return
		memcpy(target_ptr,base_ptr,num_elmts*elmt_size);
		return Event::NO_EVENT;
	}
    }

    void RegionInstanceImpl::trigger(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	memcpy(target_ptr,base_ptr,num_elmts*elmt_size);		
	complete->trigger();	
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    RegionInstanceUntyped RegionInstanceImpl::get_instance(void) const
    {
	RegionInstanceUntyped inst;
	inst.id = index;
	return inst;
    }

    Lock RegionInstanceImpl::get_lock(void)
    {
	return lock->get_lock();
    }

    void RegionInstanceAccessorUntyped<AccessorGeneric>::get_untyped(unsigned ptr_value, void *dst, size_t size) const
    {
      const void *src = ((RegionInstanceImpl *)internal_data)->read(ptr_value);
      memcpy(dst, src, size);
    }

    void RegionInstanceAccessorUntyped<AccessorGeneric>::put_untyped(unsigned ptr_value, const void *src, size_t size) const
    {
      ((RegionInstanceImpl *)internal_data)->write(ptr_value, src);
    }

    template <>
    bool RegionInstanceAccessorUntyped<AccessorGeneric>::can_convert<AccessorGeneric>(void) const
    { return true; }
    
    template <>
    RegionInstanceAccessorUntyped<AccessorGeneric> RegionInstanceAccessorUntyped<AccessorGeneric>::convert<AccessorGeneric>(void) const
    { return *this; }
    
    ////////////////////////////////////////////////////////
    // RegionMetaDataUntyped 
    ////////////////////////////////////////////////////////

    class RegionMetaDataImpl {
    public:
	RegionMetaDataImpl(int idx, Memory m, size_t num, size_t elem_size, bool activate = false) {
		PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
		active = activate;
		index = idx;
		if (activate)
		{
			num_elmts = num;
			elmt_size = elem_size;
			RegionAllocatorImpl *allocator = Runtime::get_runtime()->get_free_allocator(0,num_elmts);
			master_allocator = allocator->get_allocator();	
		
			RegionInstanceImpl* instance = Runtime::get_runtime()->get_free_instance(m,num_elmts,elmt_size);
			master_instance = instance->get_instance();
			lock = Runtime::get_runtime()->get_free_lock();
		}
	}
    public:
	bool activate(Memory m, size_t num_elmts, size_t elmt_size);
	void deactivate(void);	
	RegionMetaDataUntyped get_metadata(void);

	RegionAllocatorUntyped create_allocator(Memory m);
	RegionInstanceUntyped  create_instance(Memory m);

	void destroy_allocator(RegionAllocatorUntyped a);
	void destroy_instance(RegionInstanceUntyped i);

	Lock get_lock(void);

	RegionAllocatorUntyped get_master_allocator(void);
	RegionInstanceUntyped get_master_instance(void);
	
	void set_master_allocator(RegionAllocatorUntyped a);
	void set_master_instance(RegionInstanceUntyped i);	
    private:
	RegionAllocatorUntyped master_allocator;
	RegionInstanceUntyped  master_instance;
	//std::set<RegionAllocatorUntyped> allocators;
	std::set<RegionInstanceUntyped> instances;
	LockImpl *lock;
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

    RegionAllocatorUntyped RegionMetaDataUntyped::create_allocator_untyped(Memory m) const
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	return r->create_allocator(m);
    }

    RegionInstanceUntyped RegionMetaDataUntyped::create_instance_untyped(Memory m) const
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	return r->create_instance(m);
    }

    void RegionMetaDataUntyped::destroy_region_untyped(void) const
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	r->deactivate();
    }

    void RegionMetaDataUntyped::destroy_allocator_untyped(RegionAllocatorUntyped a) const
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	r->destroy_allocator(a);
    }

    void RegionMetaDataUntyped::destroy_instance_untyped(RegionInstanceUntyped i) const
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	r->destroy_instance(i);
    }

    Lock RegionMetaDataUntyped::get_lock(void) const
    {
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	return r->get_lock();
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
		RegionAllocatorImpl *allocator = Runtime::get_runtime()->get_free_allocator(0,num_elmts);
		master_allocator = allocator->get_allocator();	
		
		RegionInstanceImpl* instance = Runtime::get_runtime()->get_free_instance(m,num_elmts,elmt_size);
		master_instance = instance->get_instance();
		lock = Runtime::get_runtime()->get_free_lock();
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    void RegionMetaDataImpl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	active = false;
	num_elmts = 0;
	elmt_size = 0;
	RegionAllocatorImpl *allocator = Runtime::get_runtime()->get_allocator_impl(master_allocator);
	allocator->deactivate();
	for (std::set<RegionInstanceUntyped>::iterator it = instances.begin();
		it != instances.end(); it++)
	{
		RegionInstanceImpl *instance = Runtime::get_runtime()->get_instance_impl(*it);
		instance->deactivate();
	}	
	instances.clear();
	lock->deactivate();
	lock = NULL;
	master_allocator.id = 0;
	master_instance.id = 0;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    RegionMetaDataUntyped RegionMetaDataImpl::get_metadata(void)
    {
	RegionMetaDataUntyped meta;
	meta.id = index;
	return meta;
    }

    RegionAllocatorUntyped RegionMetaDataImpl::create_allocator(Memory m)
    {
	// We'll only over have one allocator for shared memory	
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	RegionAllocatorUntyped result = master_allocator;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    RegionInstanceUntyped RegionMetaDataImpl::create_instance(Memory m)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	RegionInstanceImpl* impl = Runtime::get_runtime()->get_free_instance(m,num_elmts, elmt_size);
	RegionInstanceUntyped inst = impl->get_instance();
	instances.insert(inst);
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return inst;
    }

    void RegionMetaDataImpl::destroy_allocator(RegionAllocatorUntyped a)
    {
	// Do nothing since we never made on in the first place
    }

    void RegionMetaDataImpl::destroy_instance(RegionInstanceUntyped inst)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	std::set<RegionInstanceUntyped>::iterator it = instances.find(inst);
#ifdef DEBUG_LOW_LEVEL
	assert(it != instances.end());
#endif	
	instances.erase(it);
	RegionInstanceImpl *impl = Runtime::get_runtime()->get_instance_impl(inst);
	impl->deactivate();
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    Lock RegionMetaDataImpl::get_lock(void)
    {
	return lock->get_lock();
    }

    RegionAllocatorUntyped RegionMetaDataImpl::get_master_allocator(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	RegionAllocatorUntyped result = master_allocator;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    RegionInstanceUntyped RegionMetaDataImpl::get_master_instance(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
	RegionInstanceUntyped result = master_instance;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return result;
    }

    void RegionMetaDataImpl::set_master_allocator(RegionAllocatorUntyped a)
    {
	// Do nothing since there is only ever one allocator for a meta-data in shared memory
    }

    void RegionMetaDataImpl::set_master_instance(RegionInstanceUntyped inst)	
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(&mutex));
#ifdef DEBUG_LOW_LEVEL
	// Make sure it is one of our instances
	assert(instances.find(inst) != instances.end());
#endif
	master_instance = inst;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
    }

    
    ////////////////////////////////////////////////////////
    // Machine 
    ////////////////////////////////////////////////////////

    Machine::Machine(int *argc, char ***argv,
			const Processor::TaskIDTable &task_table,
			bool cps_style, Processor::TaskFuncID init_id)
    {
	// Default nobody can use task id 0 since that is the shutdown id
	if (task_table.find(0) != task_table.end())
	{
		fprintf(stderr,"Using task_id 0 in the task table is illegal!  Task_id 0 is the shutdown task\n");
		fflush(stderr);
		exit(1);
	}

#ifdef DEBUG_PRINT
	PTHREAD_SAFE_CALL(pthread_mutex_init(&debug_mutex,NULL));
#endif
	
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
	// except for processor 0 which is this thread
	for (int id=1; id<NUM_PROCS; id++)
	{
		ProcessorImpl *impl = Runtime::runtime->processors[id];
		pthread_t thread;
		PTHREAD_SAFE_CALL(pthread_create(&thread, NULL, ProcessorImpl::start, (void*)impl));
	}
	
	// If we're doing CPS style set up the inital task and run the scheduler
	if (cps_style)
	{
		Processor p;
		p.id = 0;
		p.spawn(init_id,**argv,*argc);
		// Now run the scheduler, we'll never return from this
		ProcessorImpl *impl = Runtime::runtime->processors[0];
		impl->start((void*)impl);
	}
	// Finally do the initialization for thread 0
	local_proc_id = 0;
    }

    Machine::~Machine()
    {
    }

    void Machine::run(Processor::TaskFuncID task_id /* = 0 */)
    {
      if(task_id != 0) {
	Processor p;
	p.id = 0;
	p.spawn(task_id,0,0);
      }
      // Now run the scheduler, we'll never return from this
      ProcessorImpl *impl = Runtime::runtime->processors[0];
      impl->start((void*)impl);
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
		locks.push_back(new LockImpl(i));

	for (unsigned i=0; i<BASE_METAS; i++)
	{
		Memory m;
		m.id = 0;
		metadatas.push_back(new RegionMetaDataImpl(i,m,0,0));
	}

	for (unsigned i=0; i<BASE_ALLOCATORS; i++)
		allocators.push_back(new RegionAllocatorImpl(i,0,0));	

	for (unsigned i=0; i<BASE_INSTANCES; i++)
	{
		Memory m;
		m.id = 0;
		instances.push_back(new RegionInstanceImpl(i,m,0,0));
	}

	PTHREAD_SAFE_CALL(pthread_rwlock_init(&event_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&lock_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&metadata_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&allocator_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&instance_lock,NULL));
    }

    EventImpl* Runtime::get_event_impl(Event e)
    {
	EventImpl::EventIndex i = EventImpl::get_index(e.id);
#ifdef DEBUG_LOW_LEVEL
	assert(i != 0);
	assert(i < events.size());
#endif
	return events[i];
    }

    LockImpl* Runtime::get_lock_impl(Lock l)
    {
#ifdef DEBUG_LOW_LEVEL
	assert(l.id != 0);
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
	assert(m.id != 0);
	assert(m.id < metadatas.size());
#endif
	return metadatas[m.id];
    }

    RegionAllocatorImpl* Runtime::get_allocator_impl(RegionAllocatorUntyped a)
    {
#ifdef DEBUG_LOW_LEVEL
	assert(a.id != 0);
	assert(a.id < allocators.size());
#endif
	return allocators[a.id];
    }

    RegionInstanceImpl* Runtime::get_instance_impl(RegionInstanceUntyped i)
    {
#ifdef DEBUG_LOW_LEVEL
	assert(i.id != 0);
	assert(i.id < instances.size());
#endif
	return instances[i.id];
    }

    EventImpl* Runtime::get_free_event()
    {
	PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&event_lock));
	// Iterate over the events looking for a free one
	for (unsigned i=1; i<events.size(); i++)
	{
		if (events[i]->activate())
		{
			EventImpl *result = events[i];
			PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&event_lock));
			return result;
		}
	}
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&event_lock));
	// Otherwise there are no free events so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&event_lock));
	unsigned index = events.size();
	events.push_back(new EventImpl(index, true));
	EventImpl *result = events[index];
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&event_lock));
	return result; 
    }

    LockImpl* Runtime::get_free_lock()
    {
	PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&lock_lock));
	// Iterate over the locks looking for a free one
	for (unsigned int i=1; i<locks.size(); i++)
	{
		if (locks[i]->activate())
		{
			LockImpl *result = locks[i];
			PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&lock_lock));
			return result;
		}
	}
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&lock_lock));
	// Otherwise there are no free locks so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&lock_lock));
	unsigned index = locks.size();
	locks.push_back(new LockImpl(true));
	LockImpl *result = locks[index];
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&lock_lock));	
	return result;
    }

    RegionMetaDataImpl* Runtime::get_free_metadata(Memory m, size_t num_elmts, size_t elmt_size)
    {
	PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&metadata_lock));
	for (unsigned int i=1; i<metadatas.size(); i++)
	{
		if (metadatas[i]->activate(m,num_elmts,elmt_size))
		{
			RegionMetaDataImpl *result = metadatas[i];
			PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
			return result;
		}
	}
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
	// Otherwise there are no free metadata so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&metadata_lock));
	unsigned int index = metadatas.size();
	metadatas.push_back(new RegionMetaDataImpl(index,m,num_elmts,elmt_size,true));
	RegionMetaDataImpl *result = metadatas[index];
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
	return result;
    }

    RegionAllocatorImpl* Runtime::get_free_allocator(unsigned start, unsigned end)
    {
	PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&allocator_lock));
	for (unsigned int i=1; i<allocators.size(); i++)
	{
		if (allocators[i]->activate(start,end))
		{
			RegionAllocatorImpl*result = allocators[i];
			PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&allocator_lock));
			return result;
		}
	}
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&allocator_lock));
	// Nothing free, so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&allocator_lock));
	unsigned int index = allocators.size();
	allocators.push_back(new RegionAllocatorImpl(index,start,end,true));
	RegionAllocatorImpl*result = allocators[index];
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&allocator_lock));
	return result;
    }

    RegionInstanceImpl* Runtime::get_free_instance(Memory m, size_t num_elmts, size_t elmt_size)
    {
	PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&instance_lock));
	for (unsigned int i=1; i<instances.size(); i++)
	{
		if (instances[i]->activate(m, num_elmts,elmt_size))
		{
			RegionInstanceImpl *result = instances[i];
			PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&instance_lock));
			return result;
		}
	}
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&instance_lock));
	// Nothing free so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&instance_lock));
	unsigned int index = instances.size();
	instances.push_back(new RegionInstanceImpl(index,m,num_elmts,elmt_size,true));
	RegionInstanceImpl *result = instances[index];
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&instance_lock));
	return result;
    }

  };
};
