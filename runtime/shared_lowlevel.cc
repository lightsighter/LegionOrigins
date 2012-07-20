
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
#include <errno.h>

#define BASE_EVENTS	1024	
#define BASE_LOCKS	64	
#define BASE_METAS	64
#define BASE_ALLOCATORS	64
#define BASE_INSTANCES	64

// Minimum number of tasks are processor can
// have in its queues before the scheduler is invoked
#define MIN_SCHED_TASKS	1

// The number of threads for this version
#define NUM_PROCS	4
#define NUM_UTILITY_PROCS 0
// Maximum memory in global
#define GLOBAL_MEM      4096   // (MB)	
#define LOCAL_MEM       16384  // (KB)
// Default Pthreads stack size
#define STACK_SIZE      2      // (MB) 

#ifdef DEBUG_LOW_LEVEL
#define PTHREAD_SAFE_CALL(cmd)			\
	{					\
		int ret = (cmd);		\
		if (ret != 0) {			\
			fprintf(stderr,"PTHREAD error: %s = %d (%s)\n", #cmd, ret, strerror(ret));	\
			assert(false);		\
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

// Local processor id
__thread unsigned local_proc_id;

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
      Runtime(Machine *m, const ReductionOpTable &table);
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
      RegionMetaDataImpl*  get_free_metadata(size_t num_elmts, size_t elmt_size);
      RegionMetaDataImpl*  get_free_metadata(RegionMetaDataImpl *par, const ElementMask &mask);
      RegionAllocatorImpl* get_free_allocator(RegionMetaDataImpl *owner);
      RegionInstanceImpl*  get_free_instance(RegionMetaDataUntyped r, Memory m, size_t num_elmts, 
                                              size_t elmt_size, char *ptr, const ReductionOpUntyped *redop);

      const ReductionOpUntyped* get_reduction_op(ReductionOpID redop);

      // Return events that are free
      void free_event(EventImpl *event);
    protected:
      static Runtime *runtime;
    protected:
      friend class Machine;
      ReductionOpTable redop_table;
      std::vector<EventImpl*> events;
      std::list<EventImpl*> free_events; // Keep a free list of events since this seems to dominate perf
      std::vector<LockImpl*> locks;
      std::list<LockImpl*> free_locks;
      std::vector<MemoryImpl*> memories;
      std::vector<ProcessorImpl*> processors;
      std::vector<RegionMetaDataImpl*> metadatas;
      std::list<RegionMetaDataImpl*> free_metas;
      std::vector<RegionAllocatorImpl*> allocators;
      std::list<RegionAllocatorImpl*> free_allocators;
      std::vector<RegionInstanceImpl*> instances;
      std::list<RegionInstanceImpl*> free_instances;
      Machine *machine;
      pthread_rwlock_t event_lock;
      pthread_mutex_t  free_event_lock;
      pthread_rwlock_t lock_lock;
      pthread_mutex_t  free_lock_lock;
      pthread_rwlock_t metadata_lock;
      pthread_mutex_t  free_metas_lock;
      pthread_rwlock_t allocator_lock;
      pthread_mutex_t  free_alloc_lock;
      pthread_rwlock_t instance_lock;
      pthread_mutex_t  free_inst_lock;
    };

    /* static */
    Runtime *Runtime::runtime = NULL;
    
    struct TimerStackEntry {
    public:
      int timer_kind;
      double start_time;
      double accum_child_time;
    };

    struct PerThreadTimerData {
    public:
      PerThreadTimerData(void)
      {
        thread = local_proc_id; 
        mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
        PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
      }
      ~PerThreadTimerData(void)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
        free(mutex);
      }

      unsigned thread;
      std::list<TimerStackEntry> timer_stack;
      std::map<int, double> timer_accum;
      pthread_mutex_t *mutex;
    };

    pthread_mutex_t global_timer_mutex = PTHREAD_MUTEX_INITIALIZER;
    std::vector<PerThreadTimerData*> timer_data;
    __thread PerThreadTimerData *thread_timer_data;

#ifdef DETAILED_TIMING
    /*static*/ void DetailedTimer::clear_timers(bool all_nodes /*=true*/)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(&global_timer_mutex));
      for (std::vector<PerThreadTimerData*>::iterator it = timer_data.begin();
            it != timer_data.end(); it++)
      {
        // Take each thread's data lock as well
        PTHREAD_SAFE_CALL(pthread_mutex_lock(((*it)->mutex)));
        (*it)->timer_accum.clear();
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(((*it)->mutex)));
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(&global_timer_mutex));
    }

    /*static*/ void DetailedTimer::push_timer(int timer_kind)
    {
      if (!thread_timer_data)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&global_timer_mutex));
        thread_timer_data = new PerThreadTimerData();
        timer_data.push_back(thread_timer_data);
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&global_timer_mutex));
      }

      // no lock required here - only our thread touches the stack
      TimerStackEntry entry;
      entry.timer_kind = timer_kind;
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      entry.start_time = (1.0 * ts.tv_sec + 1e-9 * ts.tv_nsec);
      entry.accum_child_time = 0;
      thread_timer_data->timer_stack.push_back(entry);
    }

    /*static*/ void DetailedTimer::pop_timer(void)
    {
      if (!thread_timer_data)
      {
        printf("Got pop without initialized thread data !?\n");
        exit(1);
      }

      // no conflicts on stack
      TimerStackEntry old_top = thread_timer_data->timer_stack.back();
      thread_timer_data->timer_stack.pop_back();

      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      double elapsed = (1.0 * ts.tv_sec + 1e-9 * ts.tv_nsec) - old_top.start_time;

      // all the elapsed time is added to the new top as child time
      if (!thread_timer_data->timer_stack.empty())
        thread_timer_data->timer_stack.back().accum_child_time += elapsed;

      // only the elapsed time minus our own child time goes into the timer accumulator
      elapsed -= old_top.accum_child_time;

      // We do need a lock to touch the accumulator
      if (old_top.timer_kind > 0)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(thread_timer_data->mutex));
        
        std::map<int,double>::iterator it = thread_timer_data->timer_accum.find(old_top.timer_kind);
        if (it != thread_timer_data->timer_accum.end())
          it->second += elapsed;
        else
          thread_timer_data->timer_accum.insert(std::make_pair<int,double>(old_top.timer_kind,elapsed));

        PTHREAD_SAFE_CALL(pthread_mutex_unlock(thread_timer_data->mutex));
      }
    }

    /*static*/ void DetailedTimer::roll_up_timers(std::map<int,double> &timers, bool local_only)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(&global_timer_mutex));

      for (std::vector<PerThreadTimerData*>::iterator it = timer_data.begin();
            it != timer_data.end(); it++)
      {
        // Take the local lock for each thread's data too
        PTHREAD_SAFE_CALL(pthread_mutex_lock(((*it)->mutex)));

        for (std::map<int,double>::iterator it2 = (*it)->timer_accum.begin();
              it2 != (*it)->timer_accum.end(); it2++)
        {
          std::map<int,double>::iterator it3 = timers.find(it2->first);
          if (it3 != timers.end())
            it3->second += it2->second;
          else
            timers.insert(*it2);
        }

        PTHREAD_SAFE_CALL(pthread_mutex_unlock(((*it)->mutex)));
      }

      PTHREAD_SAFE_CALL(pthread_mutex_unlock(&global_timer_mutex));
    }

    /*static*/ void DetailedTimer::report_timers(bool local_only /* = false*/)
    {
      std::map<int, double> timers;

      roll_up_timers(timers, local_only);

      printf("DETAILED_TIMING_SUMMARY:\n");
      for (std::map<int,double>::iterator it = timers.begin();
            it != timers.end(); it++)
      {
        printf("%12s - %7.3f s\n", stringify(it->first), it->second);
      }
      printf("END OF DETAILED TIMING SUMMARY\n");
    }
#endif
    
    

    // Any object which can be triggered should be able to triggered
    // This will include Events and Locks
    class Triggerable {
    public:
        typedef unsigned TriggerHandle;
	virtual void trigger(unsigned count = 1, TriggerHandle = 0) = 0;
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
    public:
	EventImpl(EventIndex idx, bool activate=false) 
		: index(idx)
	{
	  in_use = activate;
	  generation = 0;
	  sources = 0;
          mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
          wait_cond = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
	  PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
	  PTHREAD_SAFE_CALL(pthread_cond_init(wait_cond,NULL));
	  if (in_use)
	  {
	    // Always initialize the current event to hand out to
	    // generation + 1, so the event will have triggered
	    // when the event matches the generation
	    current.id = index;
	    current.gen = generation+1;
	    sources = 1;
#ifdef DEBUG_LOW_LEVEL
	    assert(current.exists());
#endif
          }
	}
        ~EventImpl(void)
        {
          PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
          PTHREAD_SAFE_CALL(pthread_cond_destroy(wait_cond));
          free(mutex);
          free(wait_cond);
        }
	
	// test whether an event has triggered without waiting
	bool has_triggered(EventGeneration needed_gen);
	// block until event has triggered
	void wait(EventGeneration needed_gen, bool block);
	// create an event that won't trigger until all input events have
	Event merge_events(const std::map<EventImpl*,Event> &wait_for);
	// Trigger the event
	void trigger(unsigned count = 1, TriggerHandle handle = 0);
	// Check to see if the event is active, if not activate it (return true), otherwise false
	bool activate(void);	
	// Register a dependent event, return true if event had not been triggered and was registered
	bool register_dependent(Triggerable *target, EventGeneration needed_gen, TriggerHandle handle = 0);
	// Return an event for this EventImplementation
	Event get_event();
        // Return a user event for this EventImplementation
        UserEvent get_user_event();
        // Return a barrier for this EventImplementation
        Barrier get_barrier(unsigned expected_arrivals);
        // Alter the arrival count for the barrier
        void alter_arrival_count(int delta);
    private: 
	bool in_use;
	unsigned sources;
	const EventIndex index;
	EventGeneration generation;
	// The version of the event to hand out (i.e. with generation+1)
	// so we can detect when the event has triggered with testing
	// generational equality
	Event current; 
	pthread_mutex_t *mutex;
	pthread_cond_t *wait_cond;
	std::vector<Triggerable*> triggerables;
        std::vector<TriggerHandle> trigger_handles;
    }; 

    ////////////////////////////////////////////////////////
    // Processor Impl (up here since we need it in Event) 
    ////////////////////////////////////////////////////////

    class ProcessorImpl : public Triggerable {
    public:
	ProcessorImpl(pthread_barrier_t *init, Processor::TaskIDTable table, Processor p, size_t stacksize, bool is_utility = false, unsigned num_owners = 0) :
		init_bar(init), task_table(table), proc(p), utility_proc(p),
                has_scheduler(!is_utility && (table.find(Processor::TASK_ID_PROCESSOR_IDLE) != table.end())),
                is_utility_proc(is_utility), remaining_stops(num_owners), 
                scheduler_invoked(false), util_shutdown(is_utility) /*utility processors have no utility to shut down*/
	{
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
                wait_cond = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
		PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
		PTHREAD_SAFE_CALL(pthread_cond_init(wait_cond,NULL));
                PTHREAD_SAFE_CALL(pthread_attr_init(&attr));
                PTHREAD_SAFE_CALL(pthread_attr_setstacksize(&attr,stacksize));
		shutdown = false;
		shutdown_trigger = NULL;
                idle_task_enabled = true;
	}
        ProcessorImpl(pthread_barrier_t *init, Processor::TaskIDTable table, Processor p, Processor utility, size_t stacksize) :
                init_bar(init), task_table(table), proc(p), utility_proc(utility),
                has_scheduler(table.find(Processor::TASK_ID_PROCESSOR_IDLE) != table.end()),
                is_utility_proc(false), remaining_stops(0), 
                scheduler_invoked(false), util_shutdown(false) /*might have utility processor to shutdown*/
        {
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
                wait_cond = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
                PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
		PTHREAD_SAFE_CALL(pthread_cond_init(wait_cond,NULL));
                PTHREAD_SAFE_CALL(pthread_attr_init(&attr));
                PTHREAD_SAFE_CALL(pthread_attr_setstacksize(&attr,stacksize));
		shutdown = false;
		shutdown_trigger = NULL;
                idle_task_enabled = true;
        }
        ~ProcessorImpl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                PTHREAD_SAFE_CALL(pthread_cond_destroy(wait_cond));
                PTHREAD_SAFE_CALL(pthread_attr_destroy(&attr));
                free(mutex);
                free(wait_cond);
        }
    public:
        // Operations for utility processors
        Processor get_utility_processor(void) const;
        void release_user(Processor owner);
        void utility_finish(void);
    public:
	Event spawn(Processor::TaskFuncID func_id, const void * args,
				size_t arglen, Event wait_on);
        void run(void);
	void trigger(unsigned count = 1, TriggerHandle handle = 0);
	static void* start(void *proc);
	void preempt(EventImpl *event, EventImpl::EventGeneration needed);
    public:
        void enable_idle_task(void);
        void disable_idle_task(void);
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
    public:
        pthread_attr_t attr; // For setting pthread parameters when starting the thread
    private:
        pthread_barrier_t *init_bar;
	Processor::TaskIDTable task_table;
	Processor proc;
        Processor utility_proc;
	std::list<TaskDesc> ready_queue;
	std::list<TaskDesc> waiting_queue;
	pthread_mutex_t *mutex;
	pthread_cond_t *wait_cond;
	// Used for detecting the shutdown condition
	bool shutdown;
        bool idle_task_enabled;
	EventImpl *shutdown_trigger;
        const bool has_scheduler;
        const bool is_utility_proc;
        unsigned remaining_stops; // for utility processor knowing when to stop
        bool scheduler_invoked;   // for traking if we've invoked the scheduler
        bool util_shutdown;       // for knowing when our utility processor is done
        std::set<Processor> util_users;// Users of the utility processor to know when it's safe to finish
    };
    
    ////////////////////////////////////////////////////////
    // Events 
    ////////////////////////////////////////////////////////

    /* static */ const Event Event::NO_EVENT = { 0, 0 };

    bool Event::has_triggered(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	if (!id) return true;
	EventImpl *e = Runtime::get_runtime()->get_event_impl(*this);
	return e->has_triggered(gen);
    }

    void Event::wait(bool block) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL); 
	if (!id) return;
	EventImpl *e = Runtime::get_runtime()->get_event_impl(*this);
	e->wait(gen,block);
    }

    Event Event::merge_events(Event ev1, Event ev2, Event ev3,
                              Event ev4, Event ev5, Event ev6)
    {
      std::set<Event> wait_for;
      if (ev1.exists()) wait_for.insert(ev1);
      if (ev2.exists()) wait_for.insert(ev2);
      if (ev3.exists()) wait_for.insert(ev3);
      if (ev4.exists()) wait_for.insert(ev4);
      if (ev5.exists()) wait_for.insert(ev5);
      if (ev6.exists()) wait_for.insert(ev6);

      if (wait_for.empty())
        return Event::NO_EVENT;
      else if (wait_for.size() == 1)
        return *(wait_for.begin());
      else
        return merge_events(wait_for);
    }

    Event Event::merge_events(const std::set<Event>& wait_for)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        size_t wait_for_size = wait_for.size();
        // Ignore any no-events
        // Fast-outs for cases where there is 0 or 1 existing events
        if (wait_for.find(Event::NO_EVENT) != wait_for.end())
        {
          // Ignore the no event
          wait_for_size--;
          if (wait_for_size == 1)
          {
            Event result = Event::NO_EVENT;
            // Find the actual event
            for (std::set<Event>::const_iterator it = wait_for.begin();
                  it != wait_for.end(); it++)
            {
              result = *it;
              if (result.exists())
              {
                break;
              }
            }
#ifdef DEBUG_HIGH_LEVEL
            assert(result.exists());
#endif
            return result;
          }
        }
        else if (wait_for_size == 1)
        {
          // wait for size is 1, which means there is only one event
          Event result = *(wait_for.begin());
#ifdef DEBUG_HIGH_LEVEL
          assert(result.exists());
#endif
          return result;
        }
        // Check to make sure we have valid events
        if (wait_for_size == 0)
        {
          return Event::NO_EVENT;
        }
        // Get a new event
	EventImpl *e = Runtime::get_runtime()->get_free_event();
        // Get the implementations for all the wait_for events
        // Do this to avoid calling get_event_impl while holding the event lock
        std::map<EventImpl*,Event> wait_for_impl;
        for (std::set<Event>::const_iterator it = wait_for.begin();
              it != wait_for.end(); it++)
        {
          assert(wait_for_impl.size() < wait_for.size());
          if (!(*it).exists())
            continue;
          EventImpl *src_impl = Runtime::get_runtime()->get_event_impl(*it);
          std::pair<EventImpl*,Event> made_pair(src_impl,*it);
          wait_for_impl.insert(std::pair<EventImpl*,Event>(src_impl,*it));
        }
	return e->merge_events(wait_for_impl);
    }

    bool EventImpl::has_triggered(EventGeneration needed_gen)
    {
	bool result = false;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	result = (needed_gen <= generation);
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    void EventImpl::wait(EventGeneration needed_gen, bool block)
    {
        if (block)
        {
            // First check to see if the event has triggered
            PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));	
            // Wait until the generation indicates that the event has occurred
            while (needed_gen > generation) 
            {
                    DetailedTimer::ScopedPush sp(TIME_NONE);
                    PTHREAD_SAFE_CALL(pthread_cond_wait(wait_cond,mutex));
            }
            PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        }
        else
        {
            // Try preempting the process
            Processor local = { local_proc_id };
            ProcessorImpl *impl = Runtime::get_runtime()->get_processor_impl(local);
            // This call will only return once the event has triggered
            impl->preempt(this,needed_gen);
        }
    }

    Event EventImpl::merge_events(const std::map<EventImpl*,Event> &wait_for)
    {
	// We need the lock here so that events we've already registered
	// can't trigger this event before sources is set
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_PRINT
	//DPRINT2("Mering events into event %u generation %u\n",index,generation);
#endif
	sources = 0;
	for (std::map<EventImpl*,Event>::const_iterator it = wait_for.begin();
		it != wait_for.end(); it++)
	{
		EventImpl *src_impl = (it->first);
		// Handle the special case where this event is an older generation
		// of the same event implementation.  In this case we know it
		// already triggered.
		if (src_impl == this)
			continue;
		if (src_impl->register_dependent(this,(it->second).gen))
			sources++;
	}
	Event ret;
        // Handle the case where there are no events, or all the waiting events
        // have already triggered
        if (sources > 0)
        {
          ret = current;
        }
        else
        {
#ifdef DEBUG_LOW_LEVEL
          assert(in_use); // event should be in use
          assert(triggerables.size() == 0); // there should be no triggerables
#endif
          in_use = false;
          // return no event since all the preceding events have already triggered
          ret = Event::NO_EVENT;
        }
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        // If ret does not exist, put this back on the list of free events
        if (!ret.exists())
          Runtime::get_runtime()->free_event(this);
	return ret;
    } 

    void EventImpl::trigger(unsigned count, TriggerHandle handle)
    {
	// Update the generation
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
        assert(in_use);
	assert(sources >= count);
#endif
	sources -= count;
        bool finished = false;
	if (sources == 0)
	{
#ifdef DEBUG_PRINT
		//DPRINT2("Event %u triggered for generation %u\n",index,generation);
#endif
		// Increment the generation so that nobody can register a triggerable
		// with this event, but keep event in_use so no one can use the event
		generation++;
#ifdef DEBUG_LOW_LEVEL
		assert(generation == current.gen);
#endif
                // Wake up any waiters
		PTHREAD_SAFE_CALL(pthread_cond_broadcast(wait_cond));
		// Can't be holding the lock when triggering other triggerables
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
		// Trigger any dependent events
		while (!triggerables.empty())
		{
#ifdef DEBUG_LOW_LEVEL
                        assert(triggerables.size() == trigger_handles.size());
#endif
			triggerables.back()->trigger(1, trigger_handles.back());
			triggerables.pop_back();
                        trigger_handles.pop_back();
		}
		// Reacquire the lock and mark that in_use is false
		PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
		in_use = false;
                finished = true;
        }
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));	
        // tell the runtime that we're free
        if (finished)
          Runtime::get_runtime()->free_event(this);
    }

    bool EventImpl::activate(void)
    {
	bool result = false;
        // Try acquiring the lock, if we don't get it then just move on
#if 0
        int trythis = pthread_mutex_trylock(&mutex);
        if (trythis == EBUSY)
          return result;
        // Also check for other error codes
	PTHREAD_SAFE_CALL(trythis);
#else
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#endif
	if (!in_use)
	{
		in_use = true;
		result = true;
		sources = 1;
		// Set generation to generation+1, see 
		// comment in constructor
		current.id = index;
		current.gen = generation+1;
#ifdef DEBUG_LOW_LEVEL
		assert(current.exists());
#endif
	}	
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    bool EventImpl::register_dependent(Triggerable *target, EventGeneration gen, TriggerHandle handle)
    {
	bool result = false;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	// Make sure they're asking for the right generation, otherwise it's already triggered
	if (gen > generation)
	{
		result = true;
		// Enqueue it
		triggerables.push_back(target);	
                trigger_handles.push_back(handle);
#ifdef DEBUG_LOW_LEVEL
                assert(triggerables.size() == trigger_handles.size());
#endif
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));	
	return result;
    }

    Event EventImpl::get_event() 
    {
#ifdef DEBUG_LOW_LEVEL
        assert(in_use);
#endif
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	Event result = current;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    UserEvent EventImpl::get_user_event()
    {
#ifdef DEBUG_LOW_LEVEL
      assert(in_use);
#endif
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      UserEvent result; 
      result.id = current.id;
      result.gen = current.gen;
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      return result;
    }

    Barrier EventImpl::get_barrier(unsigned expected_arrivals)
    {
#ifdef DEBUG_LOW_LEVEL
      assert(in_use);
#endif
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      Barrier result;
      result.id = current.id;
      result.gen = current.gen;
      // Set the number of expected arrivals
      sources = expected_arrivals;
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      return result;
    }

    void EventImpl::alter_arrival_count(int delta)
    {
#ifdef DEBUG_LOW_LEVEL
      assert(in_use);
#endif
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
      if (delta < 0) // If we're deleting, make sure nothing weird happens
        assert(int(sources) > (-delta));
#endif
      sources += delta;
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    ////////////////////////////////////////////////////////
    // User Events (just use base event impl) 
    ////////////////////////////////////////////////////////

    UserEvent UserEvent::create_user_event(void)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      EventImpl *impl = Runtime::get_runtime()->get_free_event();
      return impl->get_user_event();
    }

    void UserEvent::trigger(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if (!id) return;
      EventImpl *impl = Runtime::get_runtime()->get_event_impl(*this);
      impl->trigger();
    }

    ////////////////////////////////////////////////////////
    // Barrier Events (have to use same base impl)
    ////////////////////////////////////////////////////////
    
    Barrier Barrier::create_barrier(unsigned expected_arrivals)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      EventImpl *impl = Runtime::get_runtime()->get_free_event();
      return impl->get_barrier(expected_arrivals);
    }

    void Barrier::alter_arrival_count(int delta) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if (!id) return;
      EventImpl *impl = Runtime::get_runtime()->get_event_impl(*this);
      impl->alter_arrival_count(delta);
    }

    void Barrier::arrive(unsigned count /*=1*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      if (!id) return;
      EventImpl *impl = Runtime::get_runtime()->get_event_impl(*this);
      impl->trigger(count);
    }

    ////////////////////////////////////////////////////////
    // Lock 
    ////////////////////////////////////////////////////////

    /*static*/ const Lock Lock::NO_LOCK = Lock();

    Logger::Category log_lock("lock");

    class LockImpl : public Triggerable {
    public:
	LockImpl(int idx, bool activate = false) : index(idx) {
		active = activate;
		taken = false;
		mode = 0;
		holders = 0;
		waiters = false;
                next_handle = 1;
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
	}	
        ~LockImpl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                free(mutex);
        }

	Event lock(unsigned mode, bool exclusive, Event wait_on);
	void unlock(Event wait_on);
	void trigger(unsigned count = 1, TriggerHandle handle = 0);

	bool activate(void);
	void deactivate(void);
	Lock get_lock(void) const;
    private:
	Event register_request(unsigned m, bool exc, TriggerHandle handle = 0);
	void perform_unlock(std::set<EventImpl*> &to_trigger);
    private:
	class LockRecord {
	public:
		unsigned mode;
		bool exclusive;
		Event event;
		bool handled;
                bool ready; // If this lock waits on a event, see if it's ready
                TriggerHandle id; // If it's not ready this is the trigger handle
	};
    private:
	const int index;
	bool active;
	bool taken;
	bool exclusive;
	bool waiters;
	unsigned mode;
	unsigned holders;
        TriggerHandle next_handle; // all numbers >0 are lock requests, 0 is unlock trigger handle
	std::list<LockRecord> requests;
	pthread_mutex_t *mutex;
    };

    Event Lock::lock(unsigned mode, bool exclusive, Event wait_on) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	LockImpl *l = Runtime::get_runtime()->get_lock_impl(*this);
	return l->lock(mode,exclusive, wait_on);
    }

    void Lock::unlock(Event wait_on) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	LockImpl *l = Runtime::get_runtime()->get_lock_impl(*this);
	l->unlock(wait_on);
    }

    Lock Lock::create_lock(void)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	return Runtime::get_runtime()->get_free_lock()->get_lock();
    }

    void Lock::destroy_lock(void)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	LockImpl *l = Runtime::get_runtime()->get_lock_impl(*this);
	l->deactivate();
    }

    Event LockImpl::lock(unsigned m, bool exc, Event wait_on)
    {
	Event result = Event::NO_EVENT;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        log_lock(LEVEL_DEBUG,"lock request: lock=%x mode=%d excl=%d event=%x/%d count=%d",
                 index, m, exc, wait_on.id, wait_on.gen, holders); 
        // check to see if we have to wait on event first
        bool must_wait = false;
        if (wait_on.exists())
        {
          // Try registering the lock
          EventImpl *impl = Runtime::get_runtime()->get_event_impl(wait_on);
          if (impl->register_dependent(this, wait_on.gen, next_handle))
          {
            // Successfully registered with the event, register the request as asleep
            must_wait = true;
          }
        }
        if (must_wait)
        {
          result = register_request(m, exc, next_handle);
          // Increment the next handle since we used it
          next_handle++;
        }
        else // Didn't have to wait for anything do the normal thing
        {
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
        }
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    // Always called while holding the lock
    Event LockImpl::register_request(unsigned m, bool exc, TriggerHandle handle)
    {
	EventImpl *e = Runtime::get_runtime()->get_free_event();
	LockRecord req;
	req.mode = m;
	req.exclusive = exc;
	req.event = e->get_event();
	req.handled = false;
        req.id = handle;
        // If handle is 0 then the request is already awake, otherwise wait for the trigger to occur
        req.ready = (handle == 0);
	// Add this to the list of requests
	requests.push_back(req);

	// Finally set waiters to true if it's already true
	// or there are now threads waiting
	waiters = waiters || req.ready;
	
	return req.event;
    }

    void LockImpl::unlock(Event wait_on)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        log_lock(LEVEL_DEBUG,"unlock request: lock=%x mode=%d excl=%d event=%x/%d count=%d",
                 index, mode, exclusive, wait_on.id, wait_on.gen, holders);
        std::set<EventImpl*> to_trigger;
	if (wait_on.exists())
	{
		// Register this lock to be unlocked when the even triggers	
		EventImpl *e = Runtime::get_runtime()->get_event_impl(wait_on);
                // Use default handle 0 to indicate unlock event
		if (!(e->register_dependent(this,wait_on.gen)))
		{
			// The event didn't register which means it already triggered
			// so go ahead and perform the unlock operation
			perform_unlock(to_trigger);
		}	
	}
	else
	{
		// No need to wait to perform the unlock
		perform_unlock(to_trigger);		
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        // Don't perform any triggers while holding the lock's lock
        for (std::set<EventImpl*>::const_iterator it = to_trigger.begin();
              it != to_trigger.end(); it++)
        {
          (*it)->trigger();
        }
    }

    void LockImpl::trigger(unsigned count, TriggerHandle handle)
    {
        std::set<EventImpl*> to_trigger;
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        // If the trigger handle is 0 then unlock the lock, 
        // otherwise find the lock request to wake up
        if (handle == 0)
        {
          perform_unlock(to_trigger);
        }
        else
        {
          bool found = false;
          // Go through the list and mark the matching request as being ready
          for (std::list<LockRecord>::iterator it = requests.begin();
                it != requests.end(); it++)
          {
            if (it->id == handle)
            {
              found = true;
#ifdef DEBUG_LOW_LEVEL
              assert(!it->ready);
#endif
              it->ready = true;
              // Try acquiring this lock just in case it is available,
              // otherwise we can just leave this request on the queue
              if (taken)
              {
                if (!exclusive && !it->exclusive && (mode == it->mode) && !waiters)
                {
                  holders++;
                  // Trigger the event saying we have the lock
                  to_trigger.insert(Runtime::get_runtime()->get_event_impl(it->event));
                  // Remove the request
                  requests.erase(it);
                }
                else
                {
                  // There are now definitely waiters
                  waiters = true;
                }
              }
              else // Nobody else has it, grab it!
              {
                taken = true;
                exclusive = it->exclusive;
                mode = it->mode; 
                holders = 1;
                // Trigger the event saying we have the lock
                to_trigger.insert(Runtime::get_runtime()->get_event_impl(it->event));
                // Remove this request
                requests.erase(it);
#ifdef DEBUG_PRINT
                  DPRINT3("Granting lock %d in mode %d with exclusive %d\n",index,mode,exclusive);
#endif
              }
              break;
            }
          }
#ifdef DEBUG_LOW_LEVEL
          assert(found);
#endif
        }
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        // Don't perform any triggers while holding the lock's lock
        for (std::set<EventImpl*>::const_iterator it = to_trigger.begin();
              it != to_trigger.end(); it++)
        {
          (*it)->trigger();
        }
    }

    // Always called while holding the lock's mutex
    void LockImpl::perform_unlock(std::set<EventImpl*> &to_trigger)
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
                // Clean out all the handled requests
                {
                  std::list<LockRecord>::iterator it = requests.begin();
                  while (it != requests.end())
                  {
                    if (it->handled)
                      it = requests.erase(it);
                    else
                      it++;
                  }
                }
		LockRecord req;
                bool found = false;
                for (std::list<LockRecord>::iterator it = requests.begin();
                      it != requests.end(); it++)
                {
                  if (it->ready)
                  {
                    req = *it;
                    it->handled = true;
                    found = true;
                    break;
                  }
                }
                // Check to see if we found a new candidate
                if (!found)
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
                to_trigger.insert(Runtime::get_runtime()->get_event_impl(req.event));
		// If this isn't an exclusive mode, see if there are any other
		// requests with the same mode that aren't exclusive that we can handle
		if (!exclusive)
		{
			waiters = false;
			for (std::list<LockRecord>::iterator it = requests.begin();
				it != requests.end(); it++)
			{
                          if (it->ready)
                          {
				if ((it->mode == mode) && (!it->exclusive) && (!it->handled))
				{
					it->handled = true;
                                        to_trigger.insert(Runtime::get_runtime()->get_event_impl(it->event));
					holders++;
				}
				else
				{
					// There is at least one thread still waiting
					waiters = true;
				}
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
#if 0
        int trythis = pthread_mutex_trylock(&mutex);
        if (trythis == EBUSY)
          return result;
	PTHREAD_SAFE_CALL(trythis);
#endif
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (!active)
	{
		active = true;
		result = true;
		waiters = false;
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    void LockImpl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	active = false;	
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    Lock LockImpl::get_lock(void) const
    {
#ifdef DEBUG_LOWL_LEVEL
        assert(index != 0);
#endif
	Lock l = { index };
	return l;
    }

    ////////////////////////////////////////////////////////
    // Processor 
    ////////////////////////////////////////////////////////

    /*static*/ const Processor Processor::NO_PROC = { 0 };

    // Processor Impl at top due to use in event
    
    Event Processor::spawn(Processor::TaskFuncID func_id, const void * args,
				size_t arglen, Event wait_on) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	ProcessorImpl *p = Runtime::get_runtime()->get_processor_impl(*this);
	return p->spawn(func_id, args, arglen, wait_on);
    }

    Processor Processor::get_utility_processor(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        ProcessorImpl *p = Runtime::get_runtime()->get_processor_impl(*this);
        return p->get_utility_processor();
    }

    void Processor::enable_idle_task(void)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        ProcessorImpl *p = Runtime::get_runtime()->get_processor_impl(*this);
        p->enable_idle_task();
    }

    void Processor::disable_idle_task(void)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        ProcessorImpl *p = Runtime::get_runtime()->get_processor_impl(*this);
        p->disable_idle_task();
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
	Event result = task.complete->get_event();

	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (wait_on.exists())
	{
		// Try registering this processor with the event
		EventImpl *wait_impl = Runtime::get_runtime()->get_event_impl(wait_on);
		if (!wait_impl->register_dependent(this, wait_on.gen))
		{
#ifdef DEBUG_PRINT
			DPRINT2("Registering task %d on processor %d ready queue\n",func_id,proc.id);
#endif
			// Failed to register which means it is ready to execute
			ready_queue.push_back(task);
			// If it wasn't registered, then the event triggered
			// Notify the processor thread in case it is waiting
			PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
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
		PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    Processor ProcessorImpl::get_utility_processor(void) const
    {
#ifdef DEBUG_LOW_LEVEL
        assert(!is_utility_proc);
#endif
        return utility_proc;
    }

    void ProcessorImpl::release_user(Processor owner)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
      assert(remaining_stops > 0);
      assert(util_users.find(owner) == util_users.end());
#endif
      remaining_stops--;
      util_users.insert(owner); 
      // If we've had all our users released, we can shutdown
      if (remaining_stops == 0)
      {
        shutdown = true;
      }
      // Signal in case the utility processor is waiting on work
      PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    void ProcessorImpl::utility_finish(void)
    {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
      assert(!is_utility_proc);
      assert(!util_shutdown);
#endif
      // Set util shutdown to true
      util_shutdown = true;
      // send a signal in case the processor was waiting
      PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    void ProcessorImpl::enable_idle_task(void)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        idle_task_enabled = true;
        // Wake up thread so it can run the idle task
        PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    void ProcessorImpl::disable_idle_task(void)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        idle_task_enabled = false;    
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    void ProcessorImpl::run(void)
    {
        //fprintf(stdout,"This is processor %d\n",proc.id);
        //fflush(stdout);
        // Check to see if there is an initialization task
        if (!is_utility_proc && (task_table.find(Processor::TASK_ID_PROCESSOR_INIT) != task_table.end()))
        {
          Processor::TaskFuncPtr func = task_table[Processor::TASK_ID_PROCESSOR_INIT];
          func(NULL, 0, proc);
        }
        // Wait for all the processors to be ready to go
        int bar_result = pthread_barrier_wait(init_bar);
        if (bar_result == PTHREAD_BARRIER_SERIAL_THREAD)
        {
          // Free the barrier
          PTHREAD_SAFE_CALL(pthread_barrier_destroy(init_bar));
          free(init_bar);
        }
#if DEBUG_LOW_LEVEL
        else
        {
          PTHREAD_SAFE_CALL(bar_result);
        }
        init_bar = NULL;
#endif
        //fprintf(stdout,"Processor %d is starting\n",proc.id);
        //fflush(stdout);
	// Processors run forever and permit shutdowns
	while (true)
	{
		// Make sure we're holding the lock
		PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
		// This task will perform the unlock
		execute_task(true);
	}
    }

    void ProcessorImpl::preempt(EventImpl *event, EventImpl::EventGeneration needed)
    {
	// Try registering this processor with the event in case it goes to sleep
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (!(event->register_dependent(this, needed)))
	{
		// The even triggered, release the lock and return
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
		return;
	}
	
	// Run until the event has been triggered
	while (!(event->has_triggered(needed)))
	{
		// Don't permit shutdowns since there is still a task waiting
		execute_task(false);
		// Relock the task for our next attempt
		PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	}

	// Unlock and return
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    // Must always be holding the lock when calling this task
    // This task will always unlock it
    void ProcessorImpl::execute_task(bool permit_shutdown)
    {
        // Look through the waiting queue, to see if any tasks
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
	// Check to see how many tasks there are
	// If there are too few, invoke the scheduler
        // If we've been told to shutdown, never invoke the scheduler
        // Utility proc can't have an idle task
	if (!is_utility_proc && has_scheduler && idle_task_enabled && !scheduler_invoked && 
            !shutdown && (ready_queue.size()/*+waiting_queue.size()*/) < MIN_SCHED_TASKS)
	{
                // Mark that we're invoking the scheduler
                scheduler_invoked = true;
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
                Processor::TaskFuncPtr scheduler = task_table[Processor::TASK_ID_PROCESSOR_IDLE];
                scheduler(NULL, 0, proc);
		// Return from the scheduler, so we can reevaluate status
                scheduler_invoked = false;
		return;
	}
	if (ready_queue.empty())
	{	
		if (shutdown && permit_shutdown && waiting_queue.empty())
		{
                        // Check to see if we have to wait for our utility processor to finish
                        if (!util_shutdown)
                        {
                          DetailedTimer::ScopedPush sp(TIME_NONE);
                          // Wait for our utility processor to indicate that its done
                          PTHREAD_SAFE_CALL(pthread_cond_wait(wait_cond,mutex));
                        }
                        // unlock the lock, just in case someone else decides they want to tell us something
                        // to do even though we've already exited
                        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
                        // Check to see if there is a shutdown method
                        if (!is_utility_proc && (task_table.find(Processor::TASK_ID_PROCESSOR_SHUTDOWN) != task_table.end()))
                        {
                          // If there is, call the shutdown method before triggering
                          Processor::TaskFuncPtr func = task_table[Processor::TASK_ID_PROCESSOR_SHUTDOWN];
                          func(NULL, 0, proc);
                        }
                        if (!is_utility_proc)
                        {
                          shutdown_trigger->trigger();
                        }
                        else
                        {
                          // Send shutdown messages to all our users
                          for (std::set<Processor>::const_iterator it = util_users.begin();
                                it != util_users.end(); it++)
                          {
                            ProcessorImpl *orig = Runtime::get_runtime()->get_processor_impl(*it);
                            orig->utility_finish();
                          }
                        }
                        pthread_exit(NULL);	
		}
		
		// Wait until someone tells us there is work to do unless we've been told to shutdown
                if (!shutdown)
                {
                  DetailedTimer::ScopedPush sp(TIME_NONE);
                  PTHREAD_SAFE_CALL(pthread_cond_wait(wait_cond,mutex));
                }
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	}
        else if (scheduler_invoked)
        {
                // Don't allow other tasks to be run while running the idle task
                PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
                return;
        }
	else
	{
		// Pop a task off the queue and run it
		TaskDesc task = ready_queue.front();
		ready_queue.pop_front();
		PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));	
		// Check for the shutdown function
		if (task.func_id == 0)
		{
#ifdef DEBUG_LOW_LEVEL
                        assert(!is_utility_proc); // utility processors should never get a kill pill
#endif
                        shutdown = true;
                        shutdown_trigger = task.complete;
                        // Check to see if we have a utility processor, if so mark that we're done
                        // and then set the flag to indicate when the utility processor has drained
                        // its tasks
                        if (!is_utility_proc && (utility_proc != proc))
                        {
                          util_shutdown = false;
                          // Tell our utility processor to tell us when it's done
                          ProcessorImpl *util = Runtime::get_runtime()->get_processor_impl(utility_proc);
                          util->release_user(proc);
                        }
                        else
                        {
                          // We didn't have a utility processor to shutdown
                          util_shutdown = true;
                        }
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

    void ProcessorImpl::trigger(unsigned count, TriggerHandle handle)
    {
	// We're not sure which task is ready, but at least one of them is
	// so wake up the processor thread if it is waiting
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	PTHREAD_SAFE_CALL(pthread_cond_signal(wait_cond));
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
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
    
    const Memory Memory::NO_MEMORY = {0};

    class MemoryImpl {
    public:
	MemoryImpl(size_t max) 
		: max_size(max), remaining(max)
	{
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
	}
        ~MemoryImpl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                free(mutex);
        }
    public:
	size_t remaining_bytes(void);
	void* allocate_space(size_t size);
	void free_space(void *ptr, size_t size);
        size_t total_space(void) const;  
    private:
	const size_t max_size;
	size_t remaining;
	pthread_mutex_t *mutex;
    };

    size_t MemoryImpl::remaining_bytes(void) 
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	size_t result = remaining;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    void* MemoryImpl::allocate_space(size_t size)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	void *ptr = NULL;
	if (size < remaining)
	{
		remaining -= size;
		ptr = malloc(size);
#ifdef DEBUG_LOW_LEVEL
		assert(ptr != NULL);
#endif
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return ptr;
    }

    void MemoryImpl::free_space(void *ptr, size_t size)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
	assert(ptr != NULL);
#endif
	remaining += size;
	free(ptr);
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    size_t MemoryImpl::total_space(void) const
    {
      return max_size;
    }

    ////////////////////////////////////////////////////////
    // Element Masks
    ////////////////////////////////////////////////////////

    struct ElementMaskImpl {
      //int count, offset;
      int dummy;
      unsigned bits[0];

      static size_t bytes_needed(int offset, int count)
      {
	size_t need = sizeof(ElementMaskImpl) + (((count + 31) >> 5) << 2);
	return need;
      }
	
    };

    ElementMask::ElementMask(void)
      : first_element(-1), num_elements(-1), memory(Memory::NO_MEMORY), offset(-1),
	raw_data(0), first_enabled_elmt(-1), last_enabled_elmt(-1)
    {
    }

    ElementMask::ElementMask(int _num_elements, int _first_element /*= 0*/)
      : first_element(_first_element), num_elements(_num_elements), memory(Memory::NO_MEMORY), offset(-1),
        first_enabled_elmt(-1), last_enabled_elmt(-1)
    {
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = calloc(1, bytes_needed);
      //((ElementMaskImpl *)raw_data)->count = num_elements;
      //((ElementMaskImpl *)raw_data)->offset = first_element;
    }

    ElementMask::ElementMask(const ElementMask &copy_from, 
			     int _num_elements /*= -1*/, int _first_element /*= 0*/)
    {
      first_element = copy_from.first_element;
      num_elements = copy_from.num_elements;
      first_enabled_elmt = copy_from.first_enabled_elmt;
      last_enabled_elmt = copy_from.last_enabled_elmt;
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = calloc(1, bytes_needed);

      if(copy_from.raw_data) {
	memcpy(raw_data, copy_from.raw_data, bytes_needed);
      } else {
        assert(false);
      }
    }

    ElementMask& ElementMask::operator=(const ElementMask &rhs)
    {
      first_element = rhs.first_element;
      num_elements = rhs.num_elements;
      first_enabled_elmt = rhs.first_enabled_elmt;
      last_enabled_elmt = rhs.last_enabled_elmt;
      size_t bytes_needed = rhs.raw_size();
      raw_data = calloc(1, bytes_needed);
      if (rhs.raw_data)
      {
        memcpy(raw_data, rhs.raw_data, bytes_needed);
      }
      else
      {
        assert(false);
      }
      return *this;
    }

#if 0
    void ElementMask::init(int _first_element, int _num_elements, Memory _memory, off_t _offset)
    {
      first_element = _first_element;
      num_elements = _num_elements;
      memory = _memory;
      offset = _offset;
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = Runtime::get_runtime()->get_memory_impl(memory)->get_direct_ptr(offset, bytes_needed);
    }
#endif

    void ElementMask::enable(int start, int count /*= 1*/)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	//printf("ENABLE %p %d %d %d %x\n", raw_data, offset, start, count, impl->bits[0]);
	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  unsigned *ptr = &(impl->bits[pos >> 5]);
	  *ptr |= (1U << (pos & 0x1f));
	  pos++;
	}
	//printf("ENABLED %p %d %d %d %x\n", raw_data, offset, start, count, impl->bits[0]);
      } else {
	assert(0);
#if 0
	//printf("ENABLE(2) %x %d %d %d\n", memory.id, offset, start, count);
	MemoryImpl *m_impl = Runtime::get_runtime()->get_memory_impl(memory);

	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  unsigned ofs = offset + ((pos >> 5) << 2);
	  unsigned val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  //printf("ENABLED(2) %d,  %x\n", ofs, val);
	  val |= (1U << (pos & 0x1f));
	  m_impl->put_bytes(ofs, &val, sizeof(val));
	  pos++;
	}
#endif
      }
    }

    void ElementMask::disable(int start, int count /*= 1*/)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  unsigned *ptr = &(impl->bits[pos >> 5]);
	  *ptr &= ~(1U << (pos & 0x1f));
	  pos++;
	}
      } else {
	assert(0);
#if 0
	//printf("DISABLE(2) %x %d %d %d\n", memory.id, offset, start, count);
	Memory::Impl *m_impl = memory.impl();

	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  unsigned ofs = offset + ((pos >> 5) << 2);
	  unsigned val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  //printf("DISABLED(2) %d,  %x\n", ofs, val);
	  val &= ~(1U << (pos & 0x1f));
	  m_impl->put_bytes(ofs, &val, sizeof(val));
	  pos++;
	}
#endif
      }
    }

    int ElementMask::find_enabled(int count /*= 1 */)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	//printf("FIND_ENABLED %p %d %d %x\n", raw_data, first_element, count, impl->bits[0]);
	for(int pos = first_enabled_elmt; pos <= num_elements - count; pos++) {
	  int run = 0;
	  while(1) {
	    unsigned bit = ((impl->bits[pos >> 5] >> (pos & 0x1f))) & 1;
	    if(bit != 1) break;
	    pos++; run++;
	    if(run >= count) return pos - run;
	  }
	}
      } else {
	assert(0);
#if 0
	Memory::Impl *m_impl = memory.impl();
	//printf("FIND_ENABLED(2) %x %d %d %d\n", memory.id, offset, first_element, count);
	for(int pos = 0; pos <= num_elements - count; pos++) {
	  int run = 0;
	  while(1) {
	    unsigned ofs = offset + ((pos >> 5) << 2);
	    unsigned val;
	    m_impl->get_bytes(ofs, &val, sizeof(val));
	    unsigned bit = (val >> (pos & 0x1f)) & 1;
	    if(bit != 1) break;
	    pos++; run++;
	    if(run >= count) return pos - run;
	  }
	}
#endif
      }
      return -1;
    }

    int ElementMask::find_disabled(int count /*= 1 */)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	for(int pos = 0; pos <= num_elements - count; pos++) {
	  int run = 0;
	  while(1) {
	    unsigned bit = ((impl->bits[pos >> 5] >> (pos & 0x1f))) & 1;
	    if(bit != 0) break;
	    pos++; run++;
	    if(run >= count) return pos - run;
	  }
	}
      } else {
	assert(0);
      }
      return -1;
    }

    bool ElementMask::is_set(int ptr) const
    {
        ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
        unsigned bit = ((impl->bits[ptr >> 5] >> (ptr & 0x1f))) & 1;
        return (bit == 1);
    }

    size_t ElementMask::raw_size(void) const
    {
      return ElementMaskImpl::bytes_needed(offset,num_elements);
    }

    const void *ElementMask::get_raw(void) const
    {
      return raw_data;
    }

    void ElementMask::set_raw(const void *data)
    {
      assert(0);
    }

    ElementMask::Enumerator *ElementMask::enumerate_enabled(int start /*= 0*/) const
    {
      return new ElementMask::Enumerator(*this, start, 1);
    }

    ElementMask::Enumerator *ElementMask::enumerate_disabled(int start /*= 0*/) const
    {
      return new ElementMask::Enumerator(*this, start, 0);
    }

    ElementMask::Enumerator::Enumerator(const ElementMask& _mask, int _start, int _polarity)
      : mask(_mask), pos(_start), polarity(_polarity) {}

    ElementMask::Enumerator::~Enumerator(void) {}

    bool ElementMask::Enumerator::get_next(int &position, int &length)
    {
      if(mask.raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)(mask.raw_data);

	// scan until we find a bit set with the right polarity
	while(pos < mask.num_elements) {
	  int bit = ((impl->bits[pos >> 5] >> (pos & 0x1f))) & 1;
	  if(bit != polarity) {
	    pos++;
	    continue;
	  }

	  // ok, found one bit with the right polarity - now see how many
	  //  we have in a row
	  position = pos++;
	  while(pos < mask.num_elements) {
	    int bit = ((impl->bits[pos >> 5] >> (pos & 0x1f))) & 1;
	    if(bit != polarity) break;
            pos++;
	  }
	  // we get here either because we found the end of the run or we 
	  //  hit the end of the mask
	  length = pos - position;
	  return true;
	}

	// if we fall off the end, there's no more ranges to enumerate
	return false;
      } else {
	assert(0);
#if 0
	Memory::Impl *m_impl = mask.memory.impl();

	// scan until we find a bit set with the right polarity
	while(pos < mask.num_elements) {
	  unsigned ofs = mask.offset + ((pos >> 5) << 2);
	  unsigned val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  int bit = ((val >> (pos & 0x1f))) & 1;
	  if(bit != polarity) {
	    pos++;
	    continue;
	  }

	  // ok, found one bit with the right polarity - now see how many
	  //  we have in a row
	  position = pos++;
	  while(pos < mask.num_elements) {
	    unsigned ofs = mask.offset + ((pos >> 5) << 2);
	    unsigned val;
	    m_impl->get_bytes(ofs, &val, sizeof(val));
	    int bit = ((val >> (pos & 0x1f))) & 1;
	    if(bit == polarity) {
	      pos++;
	      continue;
	    }
	  }
	  // we get here either because we found the end of the run or we 
	  //  hit the end of the mask
	  length = pos - position;
	  return true;
	}
#endif

	// if we fall off the end, there's no more ranges to enumerate
	return false;
      }
    }


    ////////////////////////////////////////////////////////
    // RegionMetaDataImpl (Declaration Only) 
    ////////////////////////////////////////////////////////

    class RegionMetaDataImpl {
    public:
	RegionMetaDataImpl(int idx, size_t num, size_t elem_size, bool activate = false) {
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
		active = activate;
		index = idx;
		if (activate)
		{
			num_elmts = num;
			elmt_size = elem_size;
			lock = Runtime::get_runtime()->get_free_lock();
                        mask = ElementMask(num_elmts);
                        parent = NULL;
		}
	}
        RegionMetaDataImpl(int idx, RegionMetaDataImpl *par, const ElementMask &m, bool activate = false) {
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
                PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
		active = activate;
		index = idx;
		if (activate)
		{
			num_elmts = m.get_num_elmts();
			elmt_size = par->elmt_size;
	                // Since we have a parent, use the parent's master allocator	
			lock = Runtime::get_runtime()->get_free_lock();
                        mask = m;
                        parent = par;
		}
        }
        ~RegionMetaDataImpl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                free(mutex);
        }
    public:
	bool activate(size_t num_elmts, size_t elmt_size);
        bool activate(RegionMetaDataImpl *par, const ElementMask &m);
	void deactivate(void);	
	RegionMetaDataUntyped get_metadata(void);

	RegionAllocatorUntyped create_allocator(Memory m);
	RegionInstanceUntyped  create_instance(Memory m, ReductionOpID redop = 0);

	void destroy_allocator(RegionAllocatorUntyped a);
	void destroy_instance(RegionInstanceUntyped i);

	Lock get_lock(void);

        const ElementMask& get_element_mask(void);
    public:
        // Traverse up the tree to the parent region that owns the master allocator
        // Peform the operation and then update the element mask on the way back down
        unsigned allocate_space(unsigned count);
        void     free_space(unsigned ptr, unsigned count);
    private:
	//std::set<RegionAllocatorUntyped> allocators;
	std::set<RegionInstanceUntyped> instances;
	LockImpl *lock;
	pthread_mutex_t *mutex;
	bool active;
	int index;
	size_t num_elmts;
	size_t elmt_size;
        ElementMask mask;
        RegionMetaDataImpl *parent;
    };

    
    ////////////////////////////////////////////////////////
    // Region Allocator 
    ////////////////////////////////////////////////////////

    /*static*/ const RegionAllocatorUntyped RegionAllocatorUntyped::NO_ALLOC = RegionAllocatorUntyped();

    class RegionAllocatorImpl {
    public:
	RegionAllocatorImpl(int idx, bool activate = false, RegionMetaDataImpl *o = NULL) 
		: owner(o), index(idx)
	{
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
		active = activate;
		if (active)
		{
			lock = Runtime::get_runtime()->get_free_lock();
		}
	}
        ~RegionAllocatorImpl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                free(mutex);
        }
    public:
	unsigned alloc_elmt(size_t num_elmts = 1);
        void free_elmt(unsigned ptr, unsigned count);
	bool activate(RegionMetaDataImpl *owner);
	void deactivate();
	RegionAllocatorUntyped get_allocator(void) const;
	Lock get_lock(void);
    private:
        RegionMetaDataImpl *owner;
	pthread_mutex_t *mutex;
	bool active;
	LockImpl *lock;
	const int index;
    }; 

    unsigned RegionAllocatorUntyped::alloc_untyped(unsigned count /*= 1*/) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return Runtime::get_runtime()->get_allocator_impl(*this)->alloc_elmt(count);
    }

    void RegionAllocatorUntyped::free_untyped(unsigned ptr, unsigned count /*= 1 */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      Runtime::get_runtime()->get_allocator_impl(*this)->free_elmt(ptr, count);
    }

    unsigned RegionAllocatorImpl::alloc_elmt(size_t num_elmts)
    {
        // No need to hold the lock since we're just reading
        return owner->allocate_space(num_elmts);
    }

    void RegionAllocatorImpl::free_elmt(unsigned ptr, unsigned count)
    {
        // No need to hold the lock since we're just reading
        owner->free_space(ptr,count);
    }

    bool RegionAllocatorImpl::activate(RegionMetaDataImpl *own)
    {
	bool result = false;
#if 0
        int trythis = pthread_mutex_trylock(&mutex);
        if (trythis == EBUSY)
          return result;
	PTHREAD_SAFE_CALL(trythis);
#endif
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (!active)
	{
                result = true;
		active = true;
                owner = own;
		lock = Runtime::get_runtime()->get_free_lock();
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    void RegionAllocatorImpl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
        assert(active);
#endif
	active = false;
	lock->deactivate();
	lock = NULL;
        owner = NULL;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    RegionAllocatorUntyped RegionAllocatorImpl::get_allocator(void) const
    {
#ifdef DEBUG_LOW_LEVEL
        assert(active);
#endif
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
        RegionInstanceImpl(int idx, RegionMetaDataUntyped r, Memory m, size_t num, size_t elem_size, 
                            bool activate = false, char *base = NULL, const ReductionOpUntyped *op = NULL)
	        : elmt_size(elem_size), num_elmts(num), reduction((op!=NULL)), redop(op), index(idx), next_handle(1)
	{
                mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
		active = activate;
		if (active)
		{
		        region = r;
			memory = m;
			// Use the memory to allocate the space, fail if there is none
			//MemoryImpl *mem = Runtime::get_runtime()->get_memory_impl(m);
			base_ptr = base; //(char*)mem->allocate_space(num_elmts*elem_size);	
#ifdef DEBUG_LOW_LEVEL
			assert(base_ptr != NULL);
#endif
			lock = Runtime::get_runtime()->get_free_lock();
		}
	}
        ~RegionInstanceImpl(void)
        {
                PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
                free(mutex);
        }
    public:
	const void* read(unsigned ptr);
	void write(unsigned ptr, const void* newval);	
        bool activate(RegionMetaDataUntyped r, Memory m, size_t num_elmts, size_t elem_size, 
                      char *base, const ReductionOpUntyped *op);
	void deactivate(void);
	Event copy_to(RegionInstanceUntyped target, Event wait_on);
        Event copy_to(RegionInstanceUntyped target, const ElementMask &mask, Event wait_on);
        Event copy_to(RegionInstanceUntyped target, RegionMetaDataUntyped src_region, Event wait_on);
	RegionInstanceUntyped get_instance(void) const;
	void trigger(unsigned count, TriggerHandle handle);
	Lock get_lock(void);
        void perform_copy_operation(RegionInstanceImpl *target, const ElementMask &src_mask, const ElementMask &dst_mask);
        bool is_reduction(void) const { return reduction; }
        void* get_base_ptr(void) const { return base_ptr; }
    private:
        class CopyOperation {
        public:
          RegionInstanceImpl *target;
          EventImpl *complete;
          TriggerHandle id;
          const ElementMask &src_mask;
          const ElementMask &dst_mask;
        public:
          CopyOperation(RegionInstanceImpl *t, EventImpl *c, TriggerHandle i, 
                        const ElementMask &s, const ElementMask &d)
            : target(t), complete(c), id(i), src_mask(s), dst_mask(d) { }
        };
    private:
        RegionMetaDataUntyped region;
	char *base_ptr;	
	size_t elmt_size;
	size_t num_elmts;
	Memory memory;
	pthread_mutex_t *mutex;
        bool reduction;
        const ReductionOpUntyped *redop;
	bool active;
	const int index;
	// Fields for the copy operation
	LockImpl *lock;
        TriggerHandle next_handle;
        std::list<CopyOperation> pending_copies;
    };

    /*static*/ const RegionInstanceUntyped RegionInstanceUntyped::NO_INST = { 0 };

   RegionInstanceAccessorUntyped<AccessorGeneric> RegionInstanceUntyped::get_accessor_untyped(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionInstanceImpl *impl = Runtime::get_runtime()->get_instance_impl(*this);
      return RegionInstanceAccessorUntyped<AccessorGeneric>((void *)impl);
    }

#if 0
    Lock RegionInstanceUntyped::get_lock(void)
    {
	return Runtime::get_runtime()->get_instance_impl(*this)->get_lock();
    }
#endif

    Event RegionInstanceUntyped::copy_to_untyped(RegionInstanceUntyped target, Event wait_on)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return Runtime::get_runtime()->get_instance_impl(*this)->copy_to(target,wait_on);
    }

    Event RegionInstanceUntyped::copy_to_untyped(RegionInstanceUntyped target, const ElementMask &mask,
                                                Event wait_on)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return Runtime::get_runtime()->get_instance_impl(*this)->copy_to(target,mask,wait_on);
    }

    Event RegionInstanceUntyped::copy_to_untyped(RegionInstanceUntyped target, RegionMetaDataUntyped region,
                                                 Event wait_on)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      return Runtime::get_runtime()->get_instance_impl(*this)->copy_to(target,region,wait_on);
    }

    const void* RegionInstanceImpl::read(unsigned ptr)
    {
      // 'ptr' has already been multiplied by elmt_size
      return ((void*)(base_ptr + ptr));
    }

    void RegionInstanceImpl::write(unsigned ptr, const void* newval)
    {
      // 'ptr' has already been multiplied by elmt_size
      memcpy((base_ptr + ptr),newval,elmt_size);
    }

    bool RegionInstanceImpl::activate(RegionMetaDataUntyped r, Memory m, size_t num, size_t elem_size, 
                                      char *base, const ReductionOpUntyped *op)
    {
	bool result = false;
#if 0
        int trythis = pthread_mutex_trylock(&mutex);
        if (trythis == EBUSY)
          return result;
	PTHREAD_SAFE_CALL(trythis);
#endif
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (!active)
	{
		active = true;
		result = true;
		region = r;
		memory = m;
		num_elmts = num;
		elmt_size = elem_size;
		//MemoryImpl *mem = Runtime::get_runtime()->get_memory_impl(m);
		base_ptr = base; //(char*)mem->allocate_space(num_elmts*elmt_size);
                redop = op;
                reduction = (redop != NULL);
#ifdef DEBUG_LOW_LEVEL
		assert(base_ptr != NULL);
#endif
		lock = Runtime::get_runtime()->get_free_lock();
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    void RegionInstanceImpl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	active = false;
	MemoryImpl *mem = Runtime::get_runtime()->get_memory_impl(memory);
	mem->free_space(base_ptr,num_elmts*elmt_size);
	num_elmts = 0;
	elmt_size = 0;
	base_ptr = NULL;	
        redop = NULL;
        reduction = false;
	lock->deactivate();
	lock = NULL;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    Logger::Category log_copy("copy");

    Event RegionInstanceImpl::copy_to(RegionInstanceUntyped target, Event wait_on)
    {
      return copy_to(target,region,wait_on);
    }

    Event RegionInstanceImpl::copy_to(RegionInstanceUntyped target, RegionMetaDataUntyped src_region, Event wait_on)
    {
      const ElementMask &mask = src_region.get_valid_mask();
      return copy_to(target,mask,wait_on);
    }

    Event RegionInstanceImpl::copy_to(RegionInstanceUntyped target, const ElementMask &mask, Event wait_on)
    {
	RegionInstanceImpl *target_impl = Runtime::get_runtime()->get_instance_impl(target);
        const ElementMask &target_mask = target_impl->region.get_valid_mask();
	//log_copy(LEVEL_INFO, "copy %x/%p/%x -> %x/%p/%x", index, this, region.id, target.id, target_impl, target_impl->region.id);
#ifdef DEBUG_LOW_LEVEL
	assert(target_impl->num_elmts == num_elmts);
	assert(target_impl->elmt_size == elmt_size);
#endif
	// Check to see if the event exists
	if (wait_on.exists())
	{
		// Try registering this as a triggerable with the event	
		EventImpl *event_impl = Runtime::get_runtime()->get_event_impl(wait_on);
		PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
		if (event_impl->register_dependent(this,wait_on.gen,next_handle))
		{
                        CopyOperation op(target_impl,Runtime::get_runtime()->get_free_event(),
                                          next_handle,mask,target_mask);
                        // Put it in the list of copy operations
                        pending_copies.push_back(op);
                        next_handle++;
			PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
			return op.complete->get_event();
		}
		else
		{
			PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
                        // Nothing to wait for
                        // Fall through and perform the copy
		}
	}
        perform_copy_operation(target_impl,mask,target_mask);
        return Event::NO_EVENT;
    }

    void RegionInstanceImpl::trigger(unsigned count, TriggerHandle handle)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        // Find the copy operation in the set
        bool found = false;
        EventImpl *complete = NULL; 
        for (std::list<CopyOperation>::iterator it = pending_copies.begin();
              it != pending_copies.end(); it++)
        {
          if (it->id == handle)
          {
            found = true;
            perform_copy_operation(it->target,it->src_mask,it->dst_mask);
            complete = it->complete;
            // Remove it from the list
            pending_copies.erase(it);
            break;
          }
        }
#ifdef DEBUG_LOW_LEVEL
        assert(found);
#endif
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        // Trigger the event saying we're done while not holding the lock!
        complete->trigger();
    }

    namespace RangeExecutors {
      class Memcpy {
      public:
        Memcpy(void *_dst_base, const void *_src_base, size_t _elmt_size)
          : dst_base((char*)_dst_base), src_base((const char*)_src_base), 
            elmt_size(_elmt_size) { }

        void do_span(int offset, int count)
        {
          off_t byte_offset = offset * elmt_size;
          size_t byte_count = count  * elmt_size;
          memcpy(dst_base + byte_offset,
                 src_base + byte_offset,
                 byte_count);
        }

      protected:
        char *dst_base;
        const char *src_base;
        size_t elmt_size;
      };

      class RedopApply {
      public:
        RedopApply(const ReductionOpUntyped *_redop, void *_dst_base,
                   const void *_src_base, size_t _elmt_size)
          : redop(_redop), dst_base((char*)_dst_base),
            src_base((const char*)_src_base), elmt_size(_elmt_size) { }

        void do_span(int offset, int count)
        {
          off_t src_offset = offset * redop->sizeof_rhs; 
          off_t dst_offset = offset * elmt_size;
          redop->apply(dst_base + dst_offset,
                       src_base + src_offset,
                       count, false/*exclusive*/);
        }

      protected:
        const ReductionOpUntyped *redop;
        char *dst_base;
        const char *src_base;
        size_t elmt_size;
      };

      class RedopFold {
      public:
        RedopFold(const ReductionOpUntyped *_redop, void *_dst_base,
                  const void *_src_base)
          : redop(_redop), dst_base((char*)_dst_base),
            src_base((const char*)_src_base) { }

        void do_span(int offset, int count)
        {
          off_t byte_offset = offset * redop->sizeof_rhs; 
          redop->fold(dst_base + byte_offset,
                      src_base + byte_offset,
                      count, false/*exclusive*/);
        }

      protected:
        const ReductionOpUntyped *redop;
        char *dst_base;
        const char *src_base;
      };
    }; // Namespace RangeExecutors

    void RegionInstanceImpl::perform_copy_operation(RegionInstanceImpl *target, const ElementMask &src_mask, const ElementMask &dst_mask)
    {
        DetailedTimer::ScopedPush sp(TIME_COPY); 
        const void *src_ptr = base_ptr;
        void       *tgt_ptr = target->base_ptr;
#ifdef DEBUG_LOW_LEVEL
        assert((src_ptr != NULL) && (tgt_ptr != NULL));
#endif
        if (!reduction)
        {
#ifdef DEBUG_LOW_LEVEL
          assert(!target->reduction);
#endif
          // This is a normal copy
          RangeExecutors::Memcpy rexec(tgt_ptr, src_ptr, elmt_size);
          ElementMask::forall_ranges(rexec, dst_mask, src_mask);
        }
        else
        {
          // This is a reduction instance, see if we are doing a reduction-to-normal copy 
          // or a reduction-to-reduction copy
          if (!target->reduction)
          {
            // Reduction-to-normal copy  
            RangeExecutors::RedopApply rexec(redop, tgt_ptr, src_ptr, elmt_size);
            ElementMask::forall_ranges(rexec, dst_mask, src_mask);
          }
          else
          {
            // Reduction-to-reduction copy
            RangeExecutors::RedopFold rexec(redop, tgt_ptr, src_ptr);
            ElementMask::forall_ranges(rexec, dst_mask, src_mask);
          }
        }
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

    void RegionInstanceAccessorUntyped<AccessorGeneric>::get_untyped(off_t byte_offset, void *dst, size_t size) const
    {
      const char *src = (const char*)(((RegionInstanceImpl *)internal_data)->get_base_ptr());
      memcpy(dst, src+byte_offset, size);
    }

    void RegionInstanceAccessorUntyped<AccessorGeneric>::put_untyped(off_t byte_offset, const void *src, size_t size) const
    {
      char *dst = (char*)(((RegionInstanceImpl *)internal_data)->get_base_ptr());
      memcpy(dst+byte_offset, src, size);
    }

    // Acessor Generic (can convert)
    template <>
    bool RegionInstanceAccessorUntyped<AccessorGeneric>::can_convert<AccessorGeneric>(void) const
    { return true; }

    template<>
    bool RegionInstanceAccessorUntyped<AccessorGeneric>::can_convert<AccessorArray>(void) const
    { 
      RegionInstanceImpl *impl = (RegionInstanceImpl*)internal_data;
      if (impl->is_reduction())
      {
        return false;
      }
      else
      {
        return true;
      }
    }

    template<>
    bool RegionInstanceAccessorUntyped<AccessorGeneric>::can_convert<AccessorArrayReductionFold>(void) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl*)internal_data;
      if (impl->is_reduction())
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    bool RegionInstanceAccessorUntyped<AccessorGeneric>::is_reduction_only(void) const
    {
      RegionInstanceImpl *impl = (RegionInstanceImpl*)internal_data;
      return impl->is_reduction();
    }

    // Accessor Generic (convert)
    template <>
    RegionInstanceAccessorUntyped<AccessorGeneric> RegionInstanceAccessorUntyped<AccessorGeneric>::convert<AccessorGeneric>(void) const
    { return *this; }

    template<>
    RegionInstanceAccessorUntyped<AccessorArray> RegionInstanceAccessorUntyped<AccessorGeneric>::convert<AccessorArray>(void) const
    { 
#ifdef DEBUG_LOW_LEVEL
      assert(!this->is_reduction_only());
#endif
      RegionInstanceImpl *impl = (RegionInstanceImpl*)internal_data;
      return RegionInstanceAccessorUntyped<AccessorArray>(impl->get_base_ptr()); 
    }

    ////////////////////////////////////////////////////////
    // RegionMetaDataUntyped 
    ////////////////////////////////////////////////////////

    /*static*/ const RegionMetaDataUntyped RegionMetaDataUntyped::NO_REGION = RegionMetaDataUntyped();

    // Lifting Declaration of RegionMetaDataImpl above allocator so we can call it in allocator
    
    Logger::Category log_region("region");

    RegionMetaDataUntyped RegionMetaDataUntyped::create_region_untyped(size_t num_elmts, size_t elmt_size)
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_free_metadata(num_elmts, elmt_size);	
	log_region(LEVEL_INFO, "region created: id=%x num=%zd size=%zd",
		   r->get_metadata().id, num_elmts, elmt_size);
	return r->get_metadata();
    }

    RegionMetaDataUntyped RegionMetaDataUntyped::create_region_untyped(RegionMetaDataUntyped parent, const ElementMask &mask)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionMetaDataImpl *par = Runtime::get_runtime()->get_metadata_impl(parent);
      RegionMetaDataImpl *r = Runtime::get_runtime()->get_free_metadata(par, mask);
      log_region(LEVEL_INFO, "region created: id=%x parent=%x",
		 r->get_metadata().id, parent.id);
      return r->get_metadata();
    }

    RegionAllocatorUntyped RegionMetaDataUntyped::create_allocator_untyped(Memory m) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	return r->create_allocator(m);
    }

    RegionInstanceUntyped RegionMetaDataUntyped::create_instance_untyped(Memory m) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	return r->create_instance(m);
    }

    RegionInstanceUntyped RegionMetaDataUntyped::create_instance_untyped(Memory m, ReductionOpID redop) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
        RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
        return r->create_instance(m, redop);
    }

    void RegionMetaDataUntyped::destroy_region_untyped(void) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
        r->deactivate();
    }

    void RegionMetaDataUntyped::destroy_allocator_untyped(RegionAllocatorUntyped a) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	r->destroy_allocator(a);
    }

    void RegionMetaDataUntyped::destroy_instance_untyped(RegionInstanceUntyped i) const
    {
        DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
	RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
	r->destroy_instance(i);
    }

    const ElementMask &RegionMetaDataUntyped::get_valid_mask(void) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      RegionMetaDataImpl *r = Runtime::get_runtime()->get_metadata_impl(*this);
      return r->get_element_mask();
    }

    bool RegionMetaDataImpl::activate(size_t num, size_t size)
    {
	bool result = false;
#if 0
        int trythis = pthread_mutex_trylock(&mutex);
        if (trythis == EBUSY)
          return result;
	PTHREAD_SAFE_CALL(trythis);
#endif
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	if (!active)
	{ 
		active = true;
		result = true;
		num_elmts = num;
		elmt_size = size;
		lock = Runtime::get_runtime()->get_free_lock();
                mask = ElementMask(num_elmts);
                parent = NULL;
	}
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
	return result;
    }

    bool RegionMetaDataImpl::activate(RegionMetaDataImpl *par, const ElementMask &m)
    {
      bool result = false;
#if 0
      int trythis = pthread_mutex_trylock(&mutex);
      if (trythis == EBUSY)
        return result;
      PTHREAD_SAFE_CALL(trythis);
#endif
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
      if (!active)
      {
        active = true;
        result = true;
        num_elmts = m.get_num_elmts();
        elmt_size = par->elmt_size;
        lock = Runtime::get_runtime()->get_free_lock();
        mask = m;
        parent = par;
      }
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
      return result;
    }

    void RegionMetaDataImpl::deactivate(void)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	active = false;
	num_elmts = 0;
	elmt_size = 0;
	for (std::set<RegionInstanceUntyped>::iterator it = instances.begin();
		it != instances.end(); it++)
	{
		RegionInstanceImpl *instance = Runtime::get_runtime()->get_instance_impl(*it);
		instance->deactivate();
	}	
	instances.clear();
	lock->deactivate();
	lock = NULL;
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    unsigned RegionMetaDataImpl::allocate_space(unsigned count)
    {
        int result = 0;
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
        if (parent == NULL)
        {
            // Do the allocation ourselves
          result = mask.find_disabled(count);
          if (result == -1)
          {
              // Allocation failure, didn't work
              fprintf(stderr,"Allocation failure in shared low level runtime. "
                  "No available space for %d elements in region %d.\n",count, index);
              exit(1);
          }
          //printf("Allocating element %d in region %d\n",result,index);
        }
        else
        {
            // Make the parent do it and intercept the returning value
            result = parent->allocate_space(count);
        }
#ifdef DEBUG_LOW_LEVEL
        assert(result >= 0);
#endif
        // Update the mask to reflect the allocation
        mask.enable(result,count);
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
        return unsigned(result);
    }

    void RegionMetaDataImpl::free_space(unsigned ptr, unsigned count)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
#ifdef DEBUG_LOW_LEVEL
        // Some sanity checks
        assert(int(ptr) < mask.get_num_elmts());
        assert(int(ptr+count) < mask.get_num_elmts());
        assert(mask.is_set(ptr));
#endif
        if (parent == NULL)
        {
           // No need to do anything here 
        }
        else
        {
            // Tell the parent to do it
            parent->free_space(ptr,count);
        }
        // Update our mask no matter what
        mask.disable(ptr,count);
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    RegionMetaDataUntyped RegionMetaDataImpl::get_metadata(void)
    {
	RegionMetaDataUntyped meta;
	meta.id = index;
	return meta;
    }

    const ElementMask& RegionMetaDataImpl::get_element_mask(void)
    {
      return mask;
    }

    RegionAllocatorUntyped RegionMetaDataImpl::create_allocator(Memory m)
    {
        RegionAllocatorImpl *allocator = Runtime::get_runtime()->get_free_allocator(this);
	return allocator->get_allocator();
    }

    RegionInstanceUntyped RegionMetaDataImpl::create_instance(Memory m, ReductionOpID redop /*=0*/)
    {
        if (!m.exists())
        {
          return RegionInstanceUntyped::NO_INST;
        }
        // First try to create the location in the memory, if there is no space
        // don't bother trying to make the data
        MemoryImpl *mem = Runtime::get_runtime()->get_memory_impl(m);
        if (redop == 0)
        {
          // No reduction op
          char *ptr = (char*)mem->allocate_space(num_elmts*elmt_size);
          if (ptr == NULL)
          {
            return RegionInstanceUntyped::NO_INST;
          }
          PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
          RegionMetaDataUntyped r = { index };
          RegionInstanceImpl* impl = Runtime::get_runtime()->get_free_instance(r,m,num_elmts, elmt_size, ptr, NULL/*redop*/);
          RegionInstanceUntyped inst = impl->get_instance();
          instances.insert(inst);
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
          return inst;
        }
        else
        {
          const ReductionOpUntyped *op = Runtime::get_runtime()->get_reduction_op(redop);
          char *ptr = (char*)mem->allocate_space(num_elmts*(op->sizeof_rhs));
          if (ptr == NULL)
          {
            return RegionInstanceUntyped::NO_INST;
          }
          // Initialize the reduction instance 
          op->init(ptr, num_elmts);
          // Set everything up
          PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
          RegionMetaDataUntyped r = { index };
          RegionInstanceImpl *impl = Runtime::get_runtime()->get_free_instance(r,m,num_elmts, op->sizeof_lhs, ptr, op);
          RegionInstanceUntyped inst = impl->get_instance();
          instances.insert(inst);
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
          return inst;
        }
    }

    void RegionMetaDataImpl::destroy_allocator(RegionAllocatorUntyped a)
    {
        RegionAllocatorImpl *allocator = Runtime::get_runtime()->get_allocator_impl(a);
        allocator->deactivate();
    }

    void RegionMetaDataImpl::destroy_instance(RegionInstanceUntyped inst)
    {
	PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
	std::set<RegionInstanceUntyped>::iterator it = instances.find(inst);
#ifdef DEBUG_LOW_LEVEL
	assert(it != instances.end());
#endif	
	instances.erase(it);
	RegionInstanceImpl *impl = Runtime::get_runtime()->get_instance_impl(inst);
	impl->deactivate();
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }

    Lock RegionMetaDataImpl::get_lock(void)
    {
	return lock->get_lock();
    }

    
    ////////////////////////////////////////////////////////
    // Machine 
    ////////////////////////////////////////////////////////

    Machine::Machine(int *argc, char ***argv,
			const Processor::TaskIDTable &task_table,
                        const ReductionOpTable &redop_table,
			bool cps_style, Processor::TaskFuncID init_id)
    {
	// Default nobody can use task id 0 since that is the shutdown id
	if (task_table.find(0) != task_table.end())
	{
		fprintf(stderr,"Using task_id 0 in the task table is illegal!  Task_id 0 is the shutdown task\n");
		fflush(stderr);
		exit(1);
	}

        unsigned num_cpus = NUM_PROCS;
        unsigned num_utility_cpus = NUM_UTILITY_PROCS;
        size_t cpu_mem_size_in_mb = GLOBAL_MEM;
        size_t cpu_l1_size_in_kb = LOCAL_MEM;
        size_t cpu_stack_size = STACK_SIZE;

#ifdef DEBUG_PRINT
	PTHREAD_SAFE_CALL(pthread_mutex_init(&debug_mutex,NULL));
#endif

        for (int i=1; i < *argc; i++)
        {
#define INT_ARG(argname, varname) do { \
	  if(!strcmp((*argv)[i], argname)) {		\
	    varname = atoi((*argv)[++i]);		\
	    continue;					\
	  } } while(0)
          
          INT_ARG("-ll:csize", cpu_mem_size_in_mb);
          INT_ARG("-ll:l1size", cpu_l1_size_in_kb);
          INT_ARG("-ll:cpu", num_cpus);
          INT_ARG("-ll:util",num_utility_cpus);
          INT_ARG("-ll:stack",cpu_stack_size);
#undef INT_ARG
        }
        cpu_stack_size = cpu_stack_size * (1 << 20);

        if (num_utility_cpus > num_cpus)
        {
            fprintf(stderr,"The number of utility cpus (%d) cannot be greater than the number of cpus (%d)\n",num_utility_cpus,num_cpus);
            fflush(stderr);
            exit(1);
        }

	// Create the runtime and initialize with this machine
	Runtime::runtime = new Runtime(this, redop_table);

        // Initialize the logger
        Logger::init(*argc, (const char**)*argv);
	
        // Keep track of the number of users of each utility cpu
        std::vector<unsigned> utility_users(num_utility_cpus);
        for (unsigned idx = 0; idx < num_utility_cpus; idx++)
        {
                utility_users[idx] = 0;
        }

	// Fill in the tables
        // find in proc 0 with NULL
        Runtime::runtime->processors.push_back(NULL);
        pthread_barrier_t *init_barrier = (pthread_barrier_t*)malloc(sizeof(pthread_barrier_t));
        PTHREAD_SAFE_CALL(pthread_barrier_init(init_barrier,NULL,(num_cpus+num_utility_cpus)));
	for (unsigned id=1; id<=num_cpus; id++)
	{
		Processor p;
		p.id = id;
		procs.insert(p);
                // Compute its utility processor (if any)
		ProcessorImpl *impl;
                if (num_utility_cpus > 0)
                {
                  unsigned util = id % num_utility_cpus;
                  Processor utility;
                  utility.id = num_cpus + 1 + util;
                  //fprintf(stdout,"Processor %d has utility processor %d\n",id,utility.id);
                  //fflush(stdout);
                  impl = new ProcessorImpl(init_barrier,task_table, p, utility, cpu_stack_size);
                  utility_users[util]++;
                }
                else
                {
                  impl = new ProcessorImpl(init_barrier,task_table, p, cpu_stack_size);
                }
		Runtime::runtime->processors.push_back(impl);
	}	
        // Also create the utility processors
        for (unsigned id=1; id<=num_utility_cpus; id++)
        {
                Processor p;
                p.id = num_cpus + id;
#ifdef DEBUG_LOW_LEVEL
                assert(utility_users[id-1] > 0);
#endif
                //fprintf(stdout,"Utility processor %d has %d users\n",p.id,utility_users[id-1]);
                //fflush(stdout);
                // This processor is a utility processor so it is be default its own utility
                ProcessorImpl *impl = new ProcessorImpl(init_barrier,task_table, p, cpu_stack_size, true/*utility*/, utility_users[id-1]);
                Runtime::runtime->processors.push_back(impl);
        }
	{
                // Make the first memory null
                Runtime::runtime->memories.push_back(NULL);
                // Do the global memory
		Memory global;
		global.id = 1;
		memories.insert(global);
		MemoryImpl *impl = new MemoryImpl(cpu_mem_size_in_mb*1024*1024);
		Runtime::runtime->memories.push_back(impl);
	}
	for (unsigned id=2; id<=(num_cpus+1); id++)
	{
		Memory m;
		m.id = id;
		memories.insert(m);
		MemoryImpl *impl = new MemoryImpl(cpu_l1_size_in_kb*1024);
		Runtime::runtime->memories.push_back(impl);
	}
	// All memories are visible from each processor
	for (unsigned id=1; id<=num_cpus; id++)
	{
		Processor p;
		p.id = id;
		visible_memories_from_procs.insert(std::pair<Processor,std::set<Memory> >(p,memories));
	}	
	// All memories are visible from all memories, all processors are visible from all memories
	for (unsigned id=1; id<=(num_cpus+1); id++)
	{
		Memory m;
		m.id = id;
		visible_memories_from_memory.insert(std::pair<Memory,std::set<Memory> >(m,memories));
		visible_procs_from_memory.insert(std::pair<Memory,std::set<Processor> >(m,procs));
	}

        // Now set up the affinities for each of the different processors and memories
        for (std::set<Processor>::iterator it = procs.begin(); it != procs.end(); it++)
        {
          // Give all processors 32 GB/s to the global memory
          {
            ProcessorMemoryAffinity global_affin = { *it, {1}, 32, 50/* higher latency */ };
            proc_mem_affinities.push_back(global_affin);
          }
          // Give the processor good affinity to its L1, but not to other L1
          for (unsigned id = 2; id <= (num_cpus+1); id++)
          {
            if (id == (it->id+1))
            {
              // Our L1, high bandwidth with low latency
              ProcessorMemoryAffinity local_affin = { *it, {id}, 100, 1/* small latency */};
              proc_mem_affinities.push_back(local_affin);
            }
            else
            {
              // Other L1, low bandwidth with long latency
              ProcessorMemoryAffinity other_affin = { *it, {id}, 10, 100 /*high latency*/ };
              proc_mem_affinities.push_back(other_affin);
            }
          }
        }
        // Set up the affinities between the different memories
        {
          // Global to all others
          for (unsigned id = 2; id <= (num_cpus+1); id++)
          {
            MemoryMemoryAffinity global_affin = { {1}, {id}, 32, 50 };
            mem_mem_affinities.push_back(global_affin);
          }

          // From any one to any other one
          for (unsigned id = 2; id <= (num_cpus+1); id++)
          {
            for (unsigned other=id+1; other <= (num_cpus+1); other++)
            {
              MemoryMemoryAffinity pair_affin = { {id}, {other}, 10, 100 };
              mem_mem_affinities.push_back(pair_affin);
            }
          }
        }
	// Now start the threads for each of the processors
	// except for processor 0 which is this thread
#ifdef DEBUG_LOW_LEVEL
        assert(Runtime::runtime->processors.size() == (num_cpus+num_utility_cpus+1));
#endif
		
	// If we're doing CPS style set up the inital task and run the scheduler
	if (cps_style)
	{
		Processor p;
		p.id = 1;
		p.spawn(init_id,**argv,*argc);
		// Now run the scheduler, we'll never return from this
		ProcessorImpl *impl = Runtime::runtime->processors[1];
		impl->start((void*)impl);
	}
	// Finally do the initialization for thread 0
	local_proc_id = 1;
    }

    Machine::~Machine()
    {
    }

    void Machine::run(Processor::TaskFuncID task_id /*= 0*/,
		      RunStyle style /*= ONE_TASK_ONLY*/,
		      const void *args /*= 0*/, size_t arglen /*= 0*/)
    {
      if(task_id != 0) { // no need to check ONE_TASK_ONLY here, since 1 node
	for(int id = 1; id <= NUM_PROCS; id++) {
	  Processor p = { id };
	  p.spawn(task_id,args,arglen);
	  if(style != ONE_TASK_PER_PROC) break;
	}
      }
      // Start the threads for each of the processors (including the utility processors)
      for (unsigned id=2; id<Runtime::runtime->processors.size(); id++)
      {
              ProcessorImpl *impl = Runtime::runtime->processors[id];
              pthread_t thread;
              PTHREAD_SAFE_CALL(pthread_create(&thread, &(impl->attr), ProcessorImpl::start, (void*)impl));
      }

      // Now run the scheduler, we'll never return from this
      ProcessorImpl *impl = Runtime::runtime->processors[1];
      ProcessorImpl::start((void*)impl);
    }

    Processor::Kind Machine::get_processor_kind(Processor p) const
    {
	return Processor::LOC_PROC;
    }

    size_t Machine::get_memory_size(const Memory m) const
    {
        return Runtime::runtime->get_memory_impl(m)->total_space();
    }

    Machine* Machine::get_machine(void)
    {
	return Runtime::get_runtime()->machine;
    }
    
    int Machine::get_proc_mem_affinity(std::vector<ProcessorMemoryAffinity> &result,
                                        Processor restrict_proc /*= Processor::NO_PROC*/,
                                        Memory restrict_memory /*= Memory::NO_MEMORY*/)
    {
      int count = 0;

      for (std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
            proc_mem_affinities.begin(); it != proc_mem_affinities.end(); it++)
      {
        if (restrict_proc.exists() && ((*it).p != restrict_proc)) continue;
        if (restrict_memory.exists() && ((*it).m != restrict_memory)) continue;
        result.push_back(*it);
        count++;
      }

      return count;
    }

    int Machine::get_mem_mem_affinity(std::vector<MemoryMemoryAffinity> &result,
                                      Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
                                      Memory restrict_mem2 /*= Memory::NO_MEMORY*/)
    {
      int count = 0;

      for (std::vector<Machine::MemoryMemoryAffinity>::const_iterator it =
            mem_mem_affinities.begin(); it != mem_mem_affinities.end(); it++)
      {
        if (restrict_mem1.exists() &&
            ((*it).m1 != restrict_mem1) && ((*it).m2 != restrict_mem1)) continue;
        if (restrict_mem2.exists() &&
            ((*it).m1 != restrict_mem2) && ((*it).m2 != restrict_mem2)) continue;
        result.push_back(*it);
        count++;
      }

      return count;
    }

    void Machine::parse_node_announce_data(const void *args, size_t arglen,
                                           const NodeAnnounceData &annc_data,
                                           bool remote)
    {
      // Should never be called in this version of the low level runtime
      assert(false);
    }

    ////////////////////////////////////////////////////////
    // Runtime 
    ////////////////////////////////////////////////////////

    Runtime::Runtime(Machine *m, const ReductionOpTable &table)
	: redop_table(table), machine(m)
    {
	for (unsigned i=0; i<BASE_EVENTS; i++)
        {
            EventImpl *event = new EventImpl(i);
            events.push_back(event);
            if (i != 0) // Don't hand out the NO_EVENT event
              free_events.push_back(event);
        }

	for (unsigned i=0; i<BASE_LOCKS; i++)
        {
		locks.push_back(new LockImpl(i));
                if (i != 0)
                  free_locks.push_back(locks.back());
        }

	for (unsigned i=0; i<BASE_METAS; i++)
	{
		metadatas.push_back(new RegionMetaDataImpl(i,0,0));
                if (i != 0)
                  free_metas.push_back(metadatas.back());
	}

	for (unsigned i=0; i<BASE_ALLOCATORS; i++)
        {
		allocators.push_back(new RegionAllocatorImpl(i));	
                if (i != 0)
                  free_allocators.push_back(allocators.back());
        }

	for (unsigned i=0; i<BASE_INSTANCES; i++)
	{
		Memory m;
		m.id = 0;
		instances.push_back(new RegionInstanceImpl(i,RegionMetaDataUntyped::NO_REGION,m,0,0));
                if (i != 0)
                  free_instances.push_back(instances.back());
	}

	PTHREAD_SAFE_CALL(pthread_rwlock_init(&event_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_event_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&lock_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_lock_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&metadata_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_metas_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&allocator_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_alloc_lock,NULL));
	PTHREAD_SAFE_CALL(pthread_rwlock_init(&instance_lock,NULL));
        PTHREAD_SAFE_CALL(pthread_mutex_init(&free_inst_lock,NULL));
    }

    EventImpl* Runtime::get_event_impl(Event e)
    {
        EventImpl::EventIndex i = e.id;
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&event_lock));
#ifdef DEBUG_LOW_LEVEL
	assert(i != 0);
	assert(i < events.size());
#endif
        EventImpl *result = events[i];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&event_lock));
	return result;
    }

    void Runtime::free_event(EventImpl *e)
    {
      // Put this event back on the list of free events
      PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_event_lock));
      free_events.push_back(e);
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_event_lock));
    }

    LockImpl* Runtime::get_lock_impl(Lock l)
    {
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&lock_lock));
#ifdef DEBUG_LOW_LEVEL
	assert(l.id != 0);
	assert(l.id < locks.size());
#endif
        LockImpl *result = locks[l.id];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&lock_lock));
	return result;
    }

    MemoryImpl* Runtime::get_memory_impl(Memory m)
    {
	if (m.id < memories.size())
		return memories[m.id];
	else
        {
                assert(false);
		return NULL;
        }
    }

    ProcessorImpl* Runtime::get_processor_impl(Processor p)
    {
#ifdef DEBUG_LOW_LEVEL
        assert(p.exists());
	assert(p.id < processors.size());
#endif
	return processors[p.id];
    }

    RegionMetaDataImpl* Runtime::get_metadata_impl(RegionMetaDataUntyped m)
    {
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&metadata_lock));
#ifdef DEBUG_LOW_LEVEL
	assert(m.id != 0);
	assert(m.id < metadatas.size());
#endif
        RegionMetaDataImpl *result = metadatas[m.id];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
	return result;
    }

    RegionAllocatorImpl* Runtime::get_allocator_impl(RegionAllocatorUntyped a)
    {
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&allocator_lock));
#ifdef DEBUG_LOW_LEVEL
	assert(a.id != 0);
	assert(a.id < allocators.size());
#endif
        RegionAllocatorImpl *result = allocators[a.id];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&allocator_lock));
	return result;
    }

    RegionInstanceImpl* Runtime::get_instance_impl(RegionInstanceUntyped i)
    {
        PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&instance_lock));
#ifdef DEBUG_LOW_LEVEL
	assert(i.id != 0);
	assert(i.id < instances.size());
#endif
        RegionInstanceImpl *result = instances[i.id];
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&instance_lock));
	return result;
    }

#if 0
    EventImpl* Runtime::get_free_event()
    {
	PTHREAD_SAFE_CALL(pthread_rwlock_rdlock(&event_lock));
	// Iterate over the events looking for a free one
	for (unsigned i=1; i<events.size(); i++)
	{
		if (events[i]->activate())
		{
			EventImpl *result = events[i];
#ifdef DEBUG_LOW_LEVEL
                        assert(result != NULL);
#endif
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
#ifdef DEBUG_LOW_LEVEL
        assert(result != NULL);
#endif
        // Create a whole bunch of other events too to avoid coming
        // into the write lock section often
        for (unsigned idx=1; idx < BASE_EVENTS; idx++)
        {
          events.push_back(new EventImpl(index+idx, false));
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&event_lock));
	return result; 
    }
#else
    EventImpl* Runtime::get_free_event()
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_event_lock));
        if (!free_events.empty())
        {
          EventImpl *result = free_events.front();
          free_events.pop_front();
          // Release the lock
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_event_lock));
          // Activate this event
          bool activated = result->activate();
#ifdef DEBUG_LOW_LEVEL
          assert(activated);
#endif
          return result;
        }
        // We weren't able to get a new event, get the writer lock
        // for the vector of event implementations and add some more
        PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&event_lock));
        unsigned index = events.size();
        EventImpl *result = new EventImpl(index,true);
        events.push_back(result);
        // Make a whole bunch of other events while we're here
        for (unsigned idx=1; idx < BASE_EVENTS; idx++)
        {
          EventImpl *temp = new EventImpl(index+idx,false);
          events.push_back(temp);
          free_events.push_back(temp);
        }
        // Release the lock on events
        PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&event_lock));
        // Release the lock on free events
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_event_lock));
        return result;
    }
#endif

    LockImpl* Runtime::get_free_lock()
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_lock_lock));
        if (!free_locks.empty())
        {
          LockImpl *result = free_locks.front();
          free_locks.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_lock_lock));
          bool activated = result->activate();
#ifdef DEBUG_LOW_LEVEL
          assert(activated);
#endif
          return result;
        }
        // We weren't able to get a new event, get the writer lock
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&lock_lock));
	unsigned index = locks.size();
	locks.push_back(new LockImpl(index,true));
	LockImpl *result = locks[index];
        // Create a whole bunch of other locks too while we're here
        for (unsigned idx=1; idx < BASE_LOCKS; idx++)
        {
          locks.push_back(new LockImpl(index+idx,false));
          free_locks.push_back(locks.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&lock_lock));	
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_lock_lock));
	return result;
    }

    RegionMetaDataImpl* Runtime::get_free_metadata(size_t num_elmts, size_t elmt_size)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_metas_lock));
        if (!free_metas.empty())
        {
          RegionMetaDataImpl *result = free_metas.front();
          free_metas.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
          bool activated = result->activate(num_elmts,elmt_size);
#ifdef DEBUG_LOW_LEVEL
          assert(activated);
#endif
          return result;
        }
	// Otherwise there are no free metadata so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&metadata_lock));
	unsigned int index = metadatas.size();
	metadatas.push_back(new RegionMetaDataImpl(index,num_elmts,elmt_size,true));
	RegionMetaDataImpl *result = metadatas[index];
        // Create a whole bunch of other metas too while we're here
        for (unsigned idx=1; idx < BASE_METAS; idx++)
        {
          metadatas.push_back(new RegionMetaDataImpl(index+idx,0,0,false));
          free_metas.push_back(metadatas.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
	return result;
    }

    RegionMetaDataImpl* Runtime::get_free_metadata(RegionMetaDataImpl *parent, const ElementMask &mask)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_metas_lock));
        if (!free_metas.empty())
        {
          RegionMetaDataImpl *result = free_metas.front();
          free_metas.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
          bool activated = result->activate(parent,mask);
#ifdef DEBUG_LOW_LEVEL
          assert(activated);
#endif
          return result;
        }
	// Otherwise there are no free metadata so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&metadata_lock));
	unsigned int index = metadatas.size();
	metadatas.push_back(new RegionMetaDataImpl(index,parent,mask,true));
	RegionMetaDataImpl *result = metadatas[index];
        // Create a whole bunch of other metas too while we're here
        for (unsigned idx=1; idx < BASE_METAS; idx++)
        {
          metadatas.push_back(new RegionMetaDataImpl(index+idx,0,0,false));
          free_metas.push_back(metadatas.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&metadata_lock));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_metas_lock));
	return result;
    }


    RegionAllocatorImpl* Runtime::get_free_allocator(RegionMetaDataImpl *owner)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_alloc_lock));
        if (!free_allocators.empty())
        {
          RegionAllocatorImpl *result = free_allocators.front();
          free_allocators.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_alloc_lock));
          bool activated = result->activate(owner);
#ifdef DEBUG_LOW_LEVEL
          assert(activated);
#endif
          return result;
        }
	// Nothing free, so make some new ones
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&allocator_lock));
	unsigned int index = allocators.size();
	allocators.push_back(new RegionAllocatorImpl(index,true,owner));
	RegionAllocatorImpl*result = allocators[index];
        // Create a whole bunch of other allocators while we're here
        for (unsigned idx=1; idx < BASE_ALLOCATORS; idx++)
        {
          allocators.push_back(new RegionAllocatorImpl(index+idx,false));
          free_allocators.push_back(allocators.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&allocator_lock));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_alloc_lock));
	return result;
    }

    RegionInstanceImpl* Runtime::get_free_instance(RegionMetaDataUntyped r, Memory m, size_t num_elmts, 
                                                    size_t elmt_size, char *ptr, const ReductionOpUntyped *redop)
    {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&free_inst_lock));
        if (!free_instances.empty())
        {
          RegionInstanceImpl *result = free_instances.front();
          free_instances.pop_front();
          PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_inst_lock));
          bool activated = result->activate(r, m, num_elmts, elmt_size, ptr, redop);
#ifdef DEBUG_LOW_LEVEL
          assert(activated);
#endif
          return result;
        }
	// Nothing free so make a new one
	PTHREAD_SAFE_CALL(pthread_rwlock_wrlock(&instance_lock));
	unsigned int index = instances.size();
	instances.push_back(new RegionInstanceImpl(index,r,m,num_elmts,elmt_size,true,ptr,redop));
	RegionInstanceImpl *result = instances[index];
        // Create a whole bunch of other instances while we're here
        for (unsigned idx=1; idx < BASE_INSTANCES; idx++)
        {
          instances.push_back(new RegionInstanceImpl(index+idx,RegionMetaDataUntyped::NO_REGION,m,0,0,false));
          free_instances.push_back(instances.back());
        }
	PTHREAD_SAFE_CALL(pthread_rwlock_unlock(&instance_lock));
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&free_inst_lock));
	return result;
    }

    const ReductionOpUntyped* Runtime::get_reduction_op(ReductionOpID redop)
    {
#ifdef DEBUG_LOW_LEVEL
      assert(redop_table.find(redop) != redop_table.end());
#endif
      return redop_table[redop];
    }

  };

  // Machine specific implementation of logvprintf
  /*static*/ void Logger::logvprintf(LogLevel level, int category, const char *fmt, va_list args)
  {
    char buffer[200];
    sprintf(buffer, "[%d - %lx] {%s}{%s}: ",
            0, /*pthread_self()*/long(local_proc_id), Logger::stringify(level), Logger::get_categories_by_id()[category].c_str());
    int len = strlen(buffer);
    vsnprintf(buffer+len, 199-len, fmt, args);
    strcat(buffer, "\n");
    fflush(stdout);
    fputs(buffer, stderr);
  }
};
