
#include "lowlevel.h"

#include <cstdio>
#include <cassert>
#include <cstdlib>

#include <map>
#include <set>
#include <list>
#include <vector>

#include <pthread.h>

#define BASE_EVENTS	64	
#define BASE_LOCKS	64	

#ifdef LOW_LEVEL_DEBUG
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
      Runtime();
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
    protected:
      static Runtime *runtime;
    private:
      std::vector<EventImpl*> events;
      std::vector<LockImpl*> locks;
      std::vector<MemoryImpl*> memories;
      std::vector<ProcessorImpl*> processors;
      std::vector<RegionMetaDataImpl*> metadatas;
      std::vector<RegionAllocatorImpl*> allocators;
      std::vector<RegionInstanceImpl*> instances;

      pthread_mutex_t mutex;
    };

    /* static */
    Runtime *Runtime::runtime = 0;

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
	EventImpl(EventIndex idx) {
	  index = idx;
	  in_use = false;
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
	LockImpl(void) {
		active = false;
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
    // Runtime 
    ////////////////////////////////////////////////////////

    Runtime::Runtime()
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
#ifdef LOW_LEVEL_DEBUG
	assert(i < events.size());
#endif
	return events[i];
    }

    LockImpl* Runtime::get_lock_impl(Lock l)
    {
#ifdef LOW_LEVEL_DEBUG
	assert(l.id < locks.size());
#endif
	return locks[l.id];
    }

    MemoryImpl* Runtime::get_memory_impl(Memory m)
    {
#ifdef LOW_LEVEL_DEBUG
	assert(m.id < memories.size());
#endif
	return memories[m.id];
    }

    ProcessorImpl* Runtime::get_processor_impl(Processor p)
    {
#ifdef LOW_LEVEL_DEBUG
	assert(p.id < processors.size());
#endif
	return processors[p.id];
    }

    RegionMetaDataImpl* Runtime::get_metadata_impl(RegionMetaDataUntyped m)
    {
#ifdef LOW_LEVEL_DEBUG
	assert(m.id < metadatas.size());
#endif
	return metadatas[m.id];
    }

    RegionAllocatorImpl* Runtime::get_allocator_impl(RegionAllocatorUntyped a)
    {
#ifdef LOW_LEVEL_DEBUG
	assert(a.id < allocators.size());
#endif
	return allocators[a.id];
    }

    RegionInstanceImpl* Runtime::get_instance_impl(RegionInstanceUntyped i)
    {
#ifdef LOW_LEVEL_DEBUG
	assert(i.id < allocator.size());
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
	events.push_back(new EventImpl(index));
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
	locks.push_back(new LockImpl());
	PTHREAD_SAFE_CALL(pthread_mutex_unlock(&mutex));
	return locks[index];
    }
  };
};
