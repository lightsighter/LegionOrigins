#include "lowlevel.h"

#include <assert.h>

#define GASNET_PAR
#include <gasnet.h>

#define GASNETT_THREAD_SAFE
#include <gasnet_tools.h>

#include "activemsg.h"

GASNETT_THREADKEY_DEFINE(in_handler);

#include <pthread.h>

#include <vector>
#include <set>
#include <list>
#include <map>

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

#define CHECK_GASNET(cmd) do { \
  int ret = (cmd); \
  if(ret != GASNET_OK) { \
    fprintf(stderr, "GASNET: %s = %d (%s, %s)\n", #cmd, ret, gasnet_ErrorName(ret), gasnet_ErrorDesc(ret)); \
    exit(1); \
  } \
} while(0)

// this is an implementation of the low level region runtime on top of GASnet+pthreads+CUDA

namespace RegionRuntime {
  namespace LowLevel {

    // for each of the ID-based runtime objects, we're going to have an
    //  implementation task and a table to look them up in
    class EventImpl;
    class LockImpl;
    class MemoryImpl;
    class ProcessorImpl;
    class RegionMetaDataImpl;
    class RegionAllocatorImpl;
    class RegionInstanceImpl;

    struct Node {
      gasnet_seginfo_t seginfo;
      std::vector<EventImpl> events;
      std::vector<LockImpl> locks;
      std::vector<MemoryImpl *> memories;
      std::vector<ProcessorImpl *> processors;
      std::vector<RegionMetaDataImpl *> metadatas;
    };

    class Runtime {
    public:
      static Runtime *get_runtime(void) { return runtime; }

      EventImpl *get_event_impl(Event e);
      LockImpl *get_lock_impl(Lock l);
      MemoryImpl *get_memory_impl(Memory m);
      ProcessorImpl *get_processor_impl(Processor p);
      RegionMetaDataImpl *get_metadata_impl(RegionMetaDataUntyped m);
      RegionAllocatorImpl *get_allocator_impl(RegionAllocatorUntyped a);
      RegionInstanceImpl *get_instance_impl(RegionInstanceUntyped i);

    protected:
    public:
      static Runtime *runtime;

      Node *nodes;
    };
    
    enum ActiveMessageIDs {
      FIRST_AVAILABLE = 128,
      NODE_ANNOUNCE_MSGID,
      SPAWN_TASK_MSGID,
      EVENT_TRIGGER_MSGID,
    };

    /*static*/ Runtime *Runtime::runtime = 0;

    ///////////////////////////////////////////////////
    // Events

    void trigger_event_handler(Event e);

    typedef ActiveMessageShortNoReply<EVENT_TRIGGER_MSGID, 
				      Event, 
				      trigger_event_handler> EventTriggerMessage;

    class EventImpl {
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

      EventImpl(void) {
	owner = 0;
	generation = 0;
	in_use = false;
	gasnet_hsl_init(&mutex);
	remote_waiters = 0;
      }

      void init(unsigned init_owner)
      {
	owner = init_owner;
      }

      // test whether an event has triggered without waiting
      bool has_triggered(EventGeneration needed_gen);

      // causes calling thread to block until event has occurred
      void wait(EventGeneration needed_gen);

      // creates an event that won't trigger until all input events have
      Event merge_events(const std::set<Event>& wait_for);

      // record that the event has triggered and notify anybody who cares
      void trigger(void);

      class EventWaiter {
      public:
	virtual void event_triggered(void) = 0;
      };

    protected:
      unsigned owner;
      EventGeneration generation;
      bool in_use;

      gasnet_hsl_t mutex; // controls which local thread has access to internal data (not runtime-visible event)

      uint64_t remote_waiters; // bitmask of which remote nodes are waiting on the event
      std::list<EventWaiter *> local_waiters; // set of local threads that are waiting on event
    };

    /*static*/ const Event Event::NO_EVENT = Event();

    bool Event::has_triggered(void) const
    {
      if(!id) return true; // special case: NO_EVENT has always triggered
      EventImpl *e = Runtime::get_runtime()->get_event_impl(*this);
      return e->has_triggered(EventImpl::get_gen(id));
    }

    void Event::wait(void) const
    {
      if(!id) return;  // special case: never wait for NO_EVENT
      EventImpl *e = Runtime::get_runtime()->get_event_impl(*this);
      e->wait(EventImpl::get_gen(id));
    }

    // creates an event that won't trigger until all input events have
    /*static*/ Event Event::merge_events(const std::set<Event>& wait_for)
    {
      assert(0);
      return Event::NO_EVENT;
    }

    /*static*/ UserEvent UserEvent::create_user_event(void)
    {
      assert(0);
      return UserEvent();
    }

    void UserEvent::trigger(void) const
    {
      Runtime::get_runtime()->get_event_impl(*this)->trigger();
    }

    void trigger_event_handler(Event e)
    {
      Runtime::get_runtime()->get_event_impl(e)->trigger();
    }

    bool EventImpl::has_triggered(EventGeneration needed_gen)
    {
      return (needed_gen < generation);
    }

    void EventImpl::wait(EventGeneration needed_gen)
    {
      assert(0);
    }
    
    void EventImpl::trigger(void)
    {
      assert(0);
    }

    ///////////////////////////////////////////////////
    // Locks

    class AutoHSLLock {
    public:
      AutoHSLLock(gasnet_hsl_t &mutex) : mutexp(&mutex) { gasnet_hsl_lock(mutexp); }
      ~AutoHSLLock(void) { gasnet_hsl_unlock(mutexp); }
    protected:
      gasnet_hsl_t *mutexp;
    };

    class LockImpl {
    public:
      LockImpl(void)
      {
	owner = 0;
	count = 0;
	mode = 0;
	gasnet_hsl_init(&mutex);
	remote_waiter_mask = 0;
	remote_sharer_mask = 0;
	requested = false;
      }

      void init(unsigned init_owner)
      {
	owner = init_owner;
      }

      class LockWaiter {
      public:
	LockWaiter(void)
	{
	  gasnett_cond_init(&condvar);
	}

	void sleep(LockImpl *impl)
	{
	  impl->local_waiters.push_back(this);
	  gasnett_cond_wait(&condvar, &impl->mutex.lock);
	}

	void wake(void)
	{
	  gasnett_cond_signal(&condvar);
	}

      protected:
	unsigned mode;
	gasnett_cond_t condvar;
      };

    protected:
      unsigned owner; // which node owns the lock
      unsigned count; // number of locks held by local threads
      unsigned mode;  // lock mode
      static const unsigned MODE_EXCL = 0;

      gasnet_hsl_t mutex; // controls which local thread has access to internal data (not runtime-visible lock)

      // bitmasks of which remote nodes are waiting on a lock (or sharing it)
      uint64_t remote_waiter_mask, remote_sharer_mask;
      std::list<LockWaiter *> local_waiters; // set of local threads that are waiting on lock
      bool requested; // do we have a request for the lock in flight?

      struct LockRequestArgs {
	gasnet_node_t node;
	unsigned lock_id;
	unsigned mode;
      };

      static void handle_lock_request(LockRequestArgs args)
      {
	assert(0);
      }

      typedef ActiveMessageShortNoReply<155, LockRequestArgs, handle_lock_request> LockRequestMessage;

      void lock(unsigned new_mode)
      {
	AutoHSLLock a(mutex); // hold mutex on lock for entire function

	while(1) { // we're going to have to loop until we succeed
	  // case 1: we own the lock
	  if(owner == gasnet_mynode()) {
	    // can we grant it?
	    if((count == 0) || ((mode == new_mode) && (mode != MODE_EXCL))) {
	      mode = new_mode;
	      count++;
	      return;
	    }
	  }

	  // case 2: we're not the owner, but we're sharing it
	  if((owner != gasnet_mynode()) && (count > 0)) {
	    // we're allowed to grant additional sharers with the same mode
	    assert(mode != MODE_EXCL);
	    if(mode == new_mode) {
	      count++;
	      return;
	    }
	  }

	  // if we fall through to here, we need to sleep until we can have
	  //  the lock - make sure to send a request to the owner node (if it's
	  //  not us, and if we haven't sent one already)
	  if((owner != gasnet_mynode()) && !requested) {
	    LockRequestArgs args;
	    args.node = gasnet_mynode();
	    args.lock_id = 44/*FIX!!!*/;
	    args.mode = new_mode;
	    LockRequestMessage::request(owner, args);
	  }

	  // now we go to sleep - somebody will wake us up when it's (probably)
	  //  our turn
	  LockWaiter waiter;
	  //waiter.mode = new_mode;
	  waiter.sleep(this);
	}
      }
    };

    Event Lock::lock(unsigned mode /* = 0 */, bool exclusive /* = true */,
		     Event wait_on /* = Event::NO_EVENT */) const
    {
      return Event::NO_EVENT;
    }

    // releases a held lock - release can be deferred until an event triggers
    void Lock::unlock(Event wait_on /* = Event::NO_EVENT */) const
    {
    }

    bool Lock::exists(void) const
    {
      return (id != 0);
    }

    // Create a new lock, destroy an existing lock
    /*static*/ Lock Lock::create_lock(void)
    {
      Lock l = { 0 };
      return l;
    }

    void Lock::destroy_lock()
    {
    }

    ///////////////////////////////////////////////////
    // Memory

    class MemoryImpl {
    public:
      MemoryImpl(size_t _size)
	: size(_size) {}

    public:
      size_t size;
    };

    ///////////////////////////////////////////////////
    // Processor

    // global because I'm being lazy...
    static Processor::TaskIDTable task_id_table;

    class ProcessorImpl {
    public:
      ProcessorImpl(Processor _me, Processor::Kind _kind)
	: me(_me), kind(_kind) {}

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      Event start_event, Event finish_event) = 0;

    public:
      Processor me;
      Processor::Kind kind;
    };

    class LocalProcessor : public ProcessorImpl {
    public:
      LocalProcessor(Processor _me, int _core_id)
	: ProcessorImpl(_me, Processor::LOC_PROC), core_id(_core_id)
      {
      }

      ~LocalProcessor(void)
      {
      }

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      Event start_event, Event finish_event)
      {
	// for now, just run the damn thing
	Processor::TaskFuncPtr fptr = task_id_table[func_id];
	Processor p;
	(*fptr)(args, arglen, p);
      }

    protected:
      int core_id;
    };

    struct SpawnTaskArgs {
      Processor proc;
      Processor::TaskFuncID func_id;
      Event start_event;
      Event finish_event;
    };

    // can't be static if it's used in a template...
    void handle_spawn_task_message(SpawnTaskArgs args,
				   const void *data, size_t datalen)
    {
      ProcessorImpl *p = Runtime::get_runtime()->get_processor_impl(args.proc);
      p->spawn_task(args.func_id, data, datalen,
		    args.start_event, args.finish_event);
    }

    typedef ActiveMessageMediumNoReply<SPAWN_TASK_MSGID,
				       SpawnTaskArgs,
				       handle_spawn_task_message> SpawnTaskMessage;

    class RemoteProcessor : public ProcessorImpl {
    public:
      RemoteProcessor(Processor _me, Processor::Kind _kind, gasnet_node_t _node)
	: ProcessorImpl(_me, _kind), node(_node)
      {
      }

      ~RemoteProcessor(void)
      {
      }

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      Event start_event, Event finish_event)
      {
	SpawnTaskArgs msgargs;
	msgargs.proc = me;
	msgargs.func_id = func_id;
	msgargs.start_event = start_event;
	msgargs.finish_event = finish_event;
	SpawnTaskMessage::request(node, msgargs, args, arglen);
      }

    protected:
      gasnet_node_t node;
    };

    Event Processor::spawn(TaskFuncID func_id, const void *args, size_t arglen,
			   Event wait_on) const
    {
      ProcessorImpl *p = Runtime::get_runtime()->get_processor_impl(*this);
      Event finish_event;
      p->spawn_task(func_id, args, arglen, wait_on, finish_event);
      return finish_event;
    }

    ///////////////////////////////////////////////////
    // Runtime

    EventImpl *Runtime::get_event_impl(Event e)
    {
      EventImpl::EventIndex i = EventImpl::get_index(e.id);
      unsigned node_id = i >> 24;
      unsigned node_ofs = i & 0xFFFFFFUL;
      Node *n = &runtime->nodes[node_id];
      if(node_ofs >= n->events.size()) {
	// grow our array to mirror additions by other nodes
	//  this should never happen for our own node
	assert(node_id != gasnet_mynode());

	unsigned oldsize = n->events.size();
	n->events.resize(node_ofs+1);
	for(unsigned i = oldsize; i <= node_ofs; i++)
	  n->events[i].init(node_id);
      }
      return &(n->events[node_ofs]);
    }

    LockImpl *Runtime::get_lock_impl(Lock l)
    {
      unsigned node_id = l.id >> 24;
      unsigned node_ofs = l.id & 0xFFFFFFUL;
      Node *n = &runtime->nodes[node_id];
      if(node_ofs >= n->locks.size()) {
	// grow our array to mirror additions by other nodes
	//  this should never happen for our own node
	assert(node_id != gasnet_mynode());

	unsigned oldsize = n->locks.size();
	n->locks.resize(node_ofs+1);
	for(unsigned i = oldsize; i <= node_ofs; i++)
	  n->locks[i].init(node_id);
      }
      return &(n->locks[node_ofs]);
    }

    MemoryImpl *Runtime::get_memory_impl(Memory p)
    {
      unsigned node_id = p.id >> 24;
      unsigned node_ofs = p.id & 0xFFFFFFUL;
      Node *n = &runtime->nodes[node_id];
      return n->memories[node_ofs];
    }

    ProcessorImpl *Runtime::get_processor_impl(Processor p)
    {
      unsigned node_id = p.id >> 24;
      unsigned node_ofs = p.id & 0xFFFFFFUL;
      Node *n = &runtime->nodes[node_id];
      return n->processors[node_ofs];
    }

    ///////////////////////////////////////////////////
    // RegionMetaData

    /*static*/ const RegionMetaDataUntyped RegionMetaDataUntyped::NO_REGION = RegionMetaDataUntyped();

    /*static*/ RegionMetaDataUntyped RegionMetaDataUntyped::create_region_untyped(Memory memory, size_t num_elmts, size_t elmt_size)
    {
      return RegionMetaDataUntyped::NO_REGION;
    }

    RegionAllocatorUntyped RegionMetaDataUntyped::create_allocator_untyped(Memory memory) const
    {
      return RegionAllocatorUntyped::NO_ALLOC;
    }

    RegionInstanceUntyped RegionMetaDataUntyped::create_instance_untyped(Memory memory) const
    {
      return RegionInstanceUntyped::NO_INST;
    }

    void RegionMetaDataUntyped::destroy_region_untyped(void) const
    {
      assert(0);
    }

    void RegionMetaDataUntyped::destroy_allocator_untyped(RegionAllocatorUntyped allocator) const
    {
      assert(0);
    }

    void RegionMetaDataUntyped::destroy_instance_untyped(RegionInstanceUntyped instance) const
    {
      assert(0);
    }

    Lock RegionMetaDataUntyped::get_lock(void) const
    {
      // we use our own ID as an ID for our lock
      Lock l = { id };
      return l;
    }

    ///////////////////////////////////////////////////
    // Region Allocators

    /*static*/ const RegionAllocatorUntyped RegionAllocatorUntyped::NO_ALLOC = RegionAllocatorUntyped();

    ///////////////////////////////////////////////////
    // Region Instances

    /*static*/ const RegionInstanceUntyped RegionInstanceUntyped::NO_INST = RegionInstanceUntyped();

    Event RegionInstanceUntyped::copy_to(RegionInstanceUntyped target, 
					 Event wait_on /*= Event::NO_EVENT*/)
    {
      assert(0);
    }

    Event RegionInstanceUntyped::copy_to(RegionInstanceUntyped target,
					 const ElementMask &mask,
					 Event wait_on /*= Event::NO_EVENT*/)
    {
      assert(0);
    }

    ///////////////////////////////////////////////////
    // 

    class ProcessorThread;

    // internal structures for locks, event, etc.
    class Task {
    public:
      typedef void(*FuncPtr)(const void *args, size_t arglen, Processor *proc);

      Task(FuncPtr _func, const void *_args, size_t _arglen,
	   ProcessorThread *_thread)
	: func(_func), arglen(_arglen), thread(_thread)
      {
	if(arglen) {
	  args = malloc(arglen);
	  memcpy(args, _args, arglen);
	} else {
	  args = 0;
	}
      }

      ~Task(void)
      {
	if(args) free(args);
      }

      void execute(Processor *proc)
      {
	(this->func)(args, arglen, proc);
      }

      FuncPtr func;
      void *args;
      size_t arglen;
      ProcessorThread *thread;
    };

    class ThreadImpl {
    public:
      ThreadImpl(void)
      {
	gasnet_hsl_init(&mutex);
	gasnett_cond_init(&condvar);
      }

      void start(void) {
	pthread_attr_t attr;
	CHECK_PTHREAD( pthread_attr_init(&attr) );
	CHECK_PTHREAD( pthread_create(&thread, &attr, &thread_main, (void *)this) );
	CHECK_PTHREAD( pthread_attr_destroy(&attr) );
      }

    protected:
      pthread_t thread;
      gasnet_hsl_t mutex;
      gasnett_cond_t condvar;

      virtual void run(void) = 0;

      static void *thread_main(void *data)
      {
	ThreadImpl *me = (ThreadImpl *) data;
	me->run();
	return 0;
      }
    };

    class ProcessorThread : public ThreadImpl {
    public:
      ProcessorThread(int _id, int _core_id)
	: id(_id), core_id(_core_id)
      {
	
      }

      void add_task(Task::FuncPtr func, const void *args, size_t arglen)
      {
	gasnet_hsl_lock(&mutex);
	pending_tasks.push_back(new Task(func, args, arglen, this));
	gasnett_cond_signal(&condvar);
	gasnet_hsl_unlock(&mutex);
      }

    protected:
      friend class LocalProcessor;
      Processor *proc;
      std::list<Task *> pending_tasks;
      int id, core_id;

      virtual void run(void)
      {
	if(core_id >= 0) {
	  cpu_set_t cset;
	  CPU_ZERO(&cset);
	  CPU_SET(core_id, &cset);
	  CHECK_PTHREAD( pthread_setaffinity_np(thread, sizeof(cset), &cset) );
	}

	printf("thread %ld running on core %d\n", thread, core_id);

	// main task loop - grab a task and run it, or sleep if no tasks
	while(1) {
	  printf("here\n"); fflush(stdout);
	  gasnet_hsl_lock(&mutex);
	  if(pending_tasks.size() > 0) {
	    Task *to_run = pending_tasks.front();
	    pending_tasks.pop_front();
	    gasnet_hsl_unlock(&mutex);

	    printf("executing task\n");
	    to_run->execute(proc);
	    delete to_run;
	  } else {
	    printf("sleeping...\n"); fflush(stdout);
	    gasnett_cond_wait(&condvar, &mutex.lock);
	    gasnet_hsl_unlock(&mutex);
	  }
	}
      }
    };

#if 0
    struct LockImpl {
    public:
      static void create_lock_table(int size)
      {
	lock_table = new LockImpl[size];
	lock_table_size = size;
      }

    protected:
      static LockImpl *lock_table;
      static int lock_table_size;

      LockImpl(void)
      {
	owner = 0;
	count = 0;
	mode = 0;
	gasnet_hsl_init(&mutex);
	remote_waiter_mask = 0;
	remote_sharer_mask = 0;
	requested = false;
      }

      class LockWaiter {
      public:
	LockWaiter(void)
	{
	  gasnett_cond_init(&condvar);
	}

	void sleep(LockImpl *impl)
	{
	  impl->local_waiters.push_back(this);
	  gasnett_cond_wait(&condvar, &impl->mutex.lock);
	}

	void wake(void)
	{
	  gasnett_cond_signal(&condvar);
	}

      protected:
	unsigned mode;
	gasnett_cond_t condvar;
      };

    protected:
      unsigned owner; // which node owns the lock
      unsigned count; // number of locks held by local threads
      unsigned mode;  // lock mode
      static const unsigned MODE_EXCL = 0;

      gasnet_hsl_t mutex; // controls which local thread has access to internal data (not runtime-visible lock)

      // bitmasks of which remote nodes are waiting on a lock (or sharing it)
      uint64_t remote_waiter_mask, remote_sharer_mask;
      std::list<LockWaiter *> local_waiters; // set of local threads that are waiting on lock
      bool requested; // do we have a request for the lock in flight?

      struct LockRequestArgs {
	gasnet_node_t node;
	unsigned lock_id;
	unsigned mode;
      };

      static void handle_lock_request(LockRequestArgs args)
      {
	assert(0);
      }

      typedef ActiveMessageShortNoReply<155, LockRequestArgs, handle_lock_request> LockRequestMessage;

      void lock(unsigned new_mode)
      {
	AutoHSLLock a(mutex); // hold mutex on lock for entire function

	while(1) { // we're going to have to loop until we succeed
	  // case 1: we own the lock
	  if(owner == gasnet_mynode()) {
	    // can we grant it?
	    if((count == 0) || ((mode == new_mode) && (mode != MODE_EXCL))) {
	      mode = new_mode;
	      count++;
	      return;
	    }
	  }

	  // case 2: we're not the owner, but we're sharing it
	  if((owner != gasnet_mynode()) && (count > 0)) {
	    // we're allowed to grant additional sharers with the same mode
	    assert(mode != MODE_EXCL);
	    if(mode == new_mode) {
	      count++;
	      return;
	    }
	  }

	  // if we fall through to here, we need to sleep until we can have
	  //  the lock - make sure to send a request to the owner node (if it's
	  //  not us, and if we haven't sent one already)
	  if((owner != gasnet_mynode()) && !requested) {
	    LockRequestArgs args;
	    args.node = gasnet_mynode();
	    args.lock_id = 44/*FIX!!!*/;
	    args.mode = new_mode;
	    LockRequestMessage::request(owner, args);
	  }

	  // now we go to sleep - somebody will wake us up when it's (probably)
	  //  our turn
	  LockWaiter waiter;
	  //waiter.mode = new_mode;
	  waiter.sleep(this);
	}
      }
    };
#endif

#if 0
    class EventImplOld {
    public:
      static void create_event_table(int size)
      {
	event_table = new EventImpl[size];
	event_table_size = size;
	// start by spreading event ownership evenly across nodes
	for(int i = 0; i < size; i++) 
	  event_table[i].owner = i % gasnet_nodes();
      }

#define MAKE_EVENT_ID(index, gen) ((((unsigned long long)(index))<<32)|((unsigned)(gen)))
#define EVENT_ID_INDEX(id) ((unsigned)((id) >> 32))
#define EVENT_ID_GEN(id)   ((unsigned)((id) & 0xFFFFFFFFUL))

      // find an available event slot and reserve it
      static Event alloc_event(void)
      {
	// simple case: see if we have an event that we own and is not in use
	for(int i = 0; i < event_table_size; i++) {
	  if((event_table[i].owner != gasnet_mynode()) ||
	     event_table[i].in_use) continue;  // no mutex - early out
	  // if we think we've got a candidate, take the lock and check for real
	  AutoHSLLock(event_table[i].mutex);
	  if((event_table[i].owner == gasnet_mynode()) && !event_table[i].in_use) {
	    event_table[i].in_use = true;
	    return Event(MAKE_EVENT_ID(i, event_table[i].generation));
	  }
	}

	// simple case failed - have to ask somebody else for one?
	assert(0); // TODO
	return Event::NO_EVENT;
      }

      static void trigger_event(uint64_t event_id);
      static void trigger_event(Event e) { trigger_event(e.event_id); }

      typedef ActiveMessageShortNoReply<144, uint64_t, EventImpl::trigger_event> EventTriggerMessage;

      class EventWaiter {
      public:
	virtual void trigger(void) = 0;
      };

    protected:
      static EventImpl *event_table;
      static int event_table_size;

      EventImpl(void)
      {
	owner = 0;
	generation = 0;
	in_use = false;
	gasnet_hsl_init(&mutex);
      }

    protected:
      unsigned owner;
      unsigned generation;
      bool in_use;

      gasnet_hsl_t mutex; // controls which local thread has access to internal data (not runtime-visible event)

      uint64_t remote_waiters; // bitmask of which remote nodes are waiting on the event
      std::list<EventWaiter *> local_waiters; // set of local threads that are waiting on event
    };

    EventImpl *EventImpl::event_table = 0;
    int EventImpl::event_table_size = 0;

    void EventImpl::trigger_event(uint64_t event_id)
      {
	EventImpl *impl = &event_table[EVENT_ID_INDEX(event_id)];
	unsigned gen = EVENT_ID_GEN(event_id);

	AutoHSLLock(impl->mutex);

	// have we already seen this trigger?  if so, drop the duplicate
	if(gen == (impl->generation - 1)) return;

	if(impl->owner == gasnet_mynode()) {
	  // if we're the owner, announce the triggering to any remote listeners
	  for(int n = 0; impl->remote_waiters; n++, impl->remote_waiters >>= 1)
	    if(impl->remote_waiters & 1)
	      EventTriggerMessage::request(n, event_id);
	} else {
	  // if we're not the owner, forward the triggering to the owner
	  EventTriggerMessage::request(impl->owner, event_id);
	}

	// in both cases, we can immediately notify any local waiters
	while(impl->local_waiters.size() > 0) {
	  EventWaiter *ew = impl->local_waiters.front();
	  impl->local_waiters.pop_front();
	  ew->trigger();
	}
      }
#endif

    // since we can't sent active messages from an active message handler,
    //   we drop them into a local circular buffer and send them out later
    class AMQueue {
    public:
      struct AMQueueEntry {
	gasnet_node_t dest;
	gasnet_handler_t handler;
	gasnet_handlerarg_t arg0, arg1, arg2, arg3;
      };

      AMQueue(unsigned _size = 1024)
	: wptr(0), rptr(0), size(_size)
      {
	gasnet_hsl_init(&mutex);
	buffer = new AMQueueEntry[_size];
      }

      ~AMQueue(void)
      {
	delete[] buffer;
      }

      void enqueue(gasnet_node_t dest, gasnet_handler_t handler,
		   gasnet_handlerarg_t arg0 = 0,
		   gasnet_handlerarg_t arg1 = 0,
		   gasnet_handlerarg_t arg2 = 0,
		   gasnet_handlerarg_t arg3 = 0)
      {
	gasnet_hsl_lock(&mutex);
	buffer[wptr].dest = dest;
	buffer[wptr].handler = handler;
	buffer[wptr].arg0 = arg0;
	buffer[wptr].arg1 = arg1;
	buffer[wptr].arg2 = arg2;
	buffer[wptr].arg3 = arg3;
	
	// now advance the write pointer - if we run into the read pointer,
	//  the world ends
	wptr = (wptr + 1) % size;
	assert(wptr != rptr);

	gasnet_hsl_unlock(&mutex);
      }

      void flush(void)
      {
	gasnet_hsl_lock(&mutex);

	while(rptr != wptr) {
	  CHECK_GASNET( gasnet_AMRequestShort4(buffer[rptr].dest,
					       buffer[rptr].handler,
					       buffer[rptr].arg0,
					       buffer[rptr].arg1,
					       buffer[rptr].arg2,
					       buffer[rptr].arg3) );
	  rptr = (rptr + 1) % size;
	}

	gasnet_hsl_unlock(&mutex);
      }

    protected:
      gasnet_hsl_t mutex;
      unsigned wptr, rptr, size;
      AMQueueEntry *buffer;
    };	

#if 0
    class LocalProcessor : public Processor {
    public:
      LocalProcessor(ProcessorThread *_thread)
	: Processor("foo"), thread(_thread)
      {
	thread->proc = this;
      }

      virtual ~LocalProcessor(void)
      {
      }

      virtual Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
			  Event wait_on = Event::NO_EVENT) const
      {
	thread->add_task(task_id_table[func_id], args, arglen);
	return Event::NO_EVENT;
      }

    protected:
      ProcessorThread *thread;
    };

    struct SpawnTaskArgs {
      unsigned proc_id;
      TaskFuncID func_id;
      Event::ID start_event;
      Event::ID finish_event;
    };

    static void spawn_task_handler(SpawnTaskArgs args, const void *data,
				   size_t datalen)
    {
    }

    typedef ActiveMessageMediumNoReply<150, SpawnTaskArgs, spawn_task_handler> SpawnTaskMessage;

    class RemoteProcessor : public Processor {
    public:
      RemoteProcessor(gasnet_node_t _node, unsigned _proc_id)
	: Processor("foo"), node(_node), proc_id(_proc_id)
      {
      }

      virtual ~RemoteProcessor(void)
      {
      }

      virtual Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
			  Event wait_on = Event::NO_EVENT) const
      {
	Event finish_event = Event::NO_EVENT;
	SpawnTaskArgs msgargs;
	msgargs.proc_id = proc_id;
	msgargs.func_id = func_id;
	msgargs.start_event = wait_on;
	msgargs.finish_event = finish_event;
	SpawnTaskMessage::request(node, msgargs, args, arglen);
	return finish_event;
      }

    protected:
      gasnet_node_t node;
      unsigned proc_id;
    };
#endif

    struct NodeAnnounceData {
      gasnet_node_t node_id;
      unsigned num_procs;
    };

    static gasnet_hsl_t announcement_mutex = GASNET_HSL_INITIALIZER;
    static int announcements_received = 0;

    void node_announce_handler(NodeAnnounceData data)
    {
      printf("%d: received announce from %d (%d procs)\n", gasnet_mynode(), data.node_id, data.num_procs);
      Node *n = &(Runtime::get_runtime()->nodes[data.node_id]);
      n->processors.resize(data.num_procs);
      for(unsigned i = 0; i < data.num_procs; i++) {
	Processor p;
	p.id = (data.node_id << 24) | i;
	n->processors[i] = new RemoteProcessor(p, Processor::LOC_PROC, data.node_id);
      }
      gasnet_hsl_lock(&announcement_mutex);
      announcements_received++;
      gasnet_hsl_unlock(&announcement_mutex);
    }

    typedef ActiveMessageShortNoReply<NODE_ANNOUNCE_MSGID,
				      NodeAnnounceData,
				      node_announce_handler> NodeAnnounceMessage;

    static void *gasnet_poll_thread_loop(void *data)
    {
      while(1) {
	gasnet_AMPoll();
	usleep(100000);
      }
      return 0;
    }

#if 0
    class GASNetNode {
    public:
      struct TestArgs {
	int x;
      };

      static void test_msg_handler(TestArgs z) { printf("got %d\n", z.x); }
      static bool test_msg_handler2(TestArgs z) { printf("got(2) %d\n", z.x); return z.x == 55; }

      typedef ActiveMessageShortNoReply<129, TestArgs, test_msg_handler> TestMessage;

      typedef ActiveMessageShortReply<133, 134, TestArgs, bool, test_msg_handler2> TestMessage2;

      GASNetNode(int *argc, char ***argv, Machine *_machine,
		 int num_local_procs = 1, int shared_mem_size = 1024)
	: machine(_machine)
      {
	CHECK_GASNET( gasnet_init(argc, argv) );
	num_nodes = gasnet_nodes();
	my_node_id = gasnet_mynode();

	gasnet_handlerentry_t handlers[128];
	int hcount = 0;
#define ADD_HANDLER(id, func) do { handlers[hcount].index = id; handlers[hcount].fnptr = (void(*)())func; hcount++; } while(0)
	ADD_HANDLER(128, am_add_task);
	hcount += TestMessage::add_handler_entries(&handlers[hcount]);
	hcount += TestMessage2::add_handler_entries(&handlers[hcount]);

	CHECK_GASNET( gasnet_attach(handlers, hcount, (shared_mem_size << 20), 0) );

	pthread_t poll_thread;
	CHECK_PTHREAD( pthread_create(&poll_thread, 0, gasnet_poll_thread_loop, 0) );
	
	gasnet_seginfo_t seginfos = new gasnet_seginfo_t[num_nodes];
	CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );

	Runtime *r = Runtime::runtime = new Runtime;
	r->nodes = new Node[num_nodes];
	for(unsigned i = 0; i < num_nodes; i++) {
	  r->nodes[i].seginfo = seginfos[i];
	}

	delete[] seginfos;

	// create local processors
	Node *n = &r->nodes[gasnet_mynode()];

	local_procs = new ProcessorThread *[num_local_procs];
	for(int i = 0; i < num_local_procs; i++) {
	  local_procs[i] = new ProcessorThread(i, -1);
	  local_procs[i]->start();
	  //machine->add_processor(new LocalProcessor(local_procs[i]));
	}

	// printf("1\n"); fflush(stdout);
	// sleep(5);
	// printf("2\n"); fflush(stdout);
	// local_procs[0]->add_task(0, 0, 0);
	// printf("3\n"); fflush(stdout);
	// sleep(5);
	// printf("4\n"); fflush(stdout);

	TestArgs zz; zz.x = 54 + my_node_id;
	TestMessage::request(0, zz);

	bool b = TestMessage2::request(0, zz);
	printf("return = %d\n", b);
      }

      ~GASNetNode(void)
      {
	gasnet_exit(0);
      }

      static GASNetNode *my_node;

    protected:
      AMQueue outgoing_ams;

      // ID:128 adds a task to a specified processor - no reply
      static void am_add_task(gasnet_token_t token,
			      void *buf, size_t nbytes,
			      gasnet_handlerarg_t proc_id,
			      gasnet_handlerarg_t func_id,
			      gasnet_handlerarg_t wait_event_id,
			      gasnet_handlerarg_t finish_event_id)
      {
	my_node->local_procs[proc_id]->add_task(task_id_table[func_id],
						//wait_event_id, finish_event_id,
						buf, nbytes);
      }

      unsigned num_nodes, my_node_id;
      ProcessorThread **local_procs;
      gasnet_seginfo_t *seginfo;
      Machine *machine;
    };

    GASNetNode *GASNetNode::my_node = 0;

#endif
    static Machine *the_machine = 0;

    /*static*/ Machine *Machine::get_machine(void) { return the_machine; }

    Machine::Machine(int *argc, char ***argv,
		     const Processor::TaskIDTable &task_table,
		     bool cps_style /* = false */,
		     Processor::TaskFuncID init_id /* = 0 */)
    {
      for(Processor::TaskIDTable::const_iterator it = task_table.begin();
	  it != task_table.end();
	  it++)
	task_id_table[it->first] = it->second;

      //GASNetNode::my_node = new GASNetNode(argc, argv, this);
      CHECK_GASNET( gasnet_init(argc, argv) );

      gasnet_handlerentry_t handlers[128];
      int hcount = 0;
      hcount += NodeAnnounceMessage::add_handler_entries(&handlers[hcount]);
      hcount += SpawnTaskMessage::add_handler_entries(&handlers[hcount]);
      hcount += EventTriggerMessage::add_handler_entries(&handlers[hcount]);
      //hcount += TestMessage::add_handler_entries(&handlers[hcount]);
      //hcount += TestMessage2::add_handler_entries(&handlers[hcount]);

      unsigned shared_mem_size = 32;
      CHECK_GASNET( gasnet_attach(handlers, hcount, (shared_mem_size << 20), 0) );

      pthread_t poll_thread;
      CHECK_PTHREAD( pthread_create(&poll_thread, 0, gasnet_poll_thread_loop, 0) );
	
      gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[gasnet_nodes()];
      CHECK_GASNET( gasnet_getSegmentInfo(seginfos, gasnet_nodes()) );

      Runtime *r = Runtime::runtime = new Runtime;
      r->nodes = new Node[gasnet_nodes()];
      for(unsigned i = 0; i < gasnet_nodes(); i++) {
	r->nodes[i].seginfo = seginfos[i];
      }
      
      delete[] seginfos;

      Node *n = &r->nodes[gasnet_mynode()];

      NodeAnnounceData announce_data;

      unsigned num_local_procs = 1;

      announce_data.node_id = gasnet_mynode();
      announce_data.num_procs = num_local_procs;

      // create local processors
      n->processors.resize(num_local_procs);

      for(unsigned i = 0; i < num_local_procs; i++) {
	Processor p;
	p.id = (gasnet_mynode() << 24) | i;
	n->processors[i] = new LocalProcessor(p, i);
	//local_procs[i]->start();
	//machine->add_processor(new LocalProcessor(local_procs[i]));
      }

      // now announce ourselves to everyone else
      for(int i = 0; i < gasnet_nodes(); i++)
	if(i != gasnet_mynode())
	  NodeAnnounceMessage::request(i, announce_data);

      // wait until we hear from everyone else?
      while(announcements_received < (gasnet_nodes() - 1))
	gasnet_AMPoll();

      printf("node %d has received all of its announcements\n", gasnet_mynode());

      for(int i = 0; i < gasnet_nodes(); i++)
	for(unsigned j = 0; j < Runtime::runtime->nodes[i].processors.size(); j++) {
	  Processor p;
	  p.id = (i << 24) | j;
	  procs.insert(p);
	}

      the_machine = this;
    }

    Machine::~Machine(void)
    {
      gasnet_exit(0);
    }

    Processor::Kind Machine::get_processor_kind(Processor p) const
    {
      ProcessorImpl *impl = Runtime::runtime->get_processor_impl(p);
      return impl->kind;
    }

    size_t Machine::get_memory_size(const Memory m) const
    {
      MemoryImpl *impl = Runtime::runtime->get_memory_impl(m);
      return impl->size;
    }

    void Machine::run(Processor::TaskFuncID task_id /* = 0*/)
    {
      assert(0);
    }


  }; // namespace LowLevel
}; // namespace RegionRuntime

// int main(int argc, const char *argv[])
// {
//   RegionRuntime::LowLevel::GASNetNode my_node(argc, (char **)argv);
//   printf("hello, world!\n");
//   printf("limits:\n");
//   printf("max args: %zd (%zd bytes each)\n", gasnet_AMMaxArgs(), sizeof(gasnet_handlerarg_t));
//   printf("max medium: %zd\n", gasnet_AMMaxMedium());
//   printf("long req: %zd\n", gasnet_AMMaxLongRequest());
//   printf("long reply: %zd\n", gasnet_AMMaxLongReply());
// }
