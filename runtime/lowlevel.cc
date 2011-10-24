#include "lowlevel.h"

#include <assert.h>

#define GASNET_PAR
#include <gasnet.h>

#define GASNETT_THREAD_SAFE
#include <gasnet_tools.h>

#include "activemsg.h"

GASNETT_THREADKEY_DEFINE(in_handler);
GASNETT_THREADKEY_DEFINE(cur_thread);

#include <pthread.h>

#include <vector>
#include <deque>
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

// GASnet helper stuff

class AutoHSLLock {
public:
  AutoHSLLock(gasnet_hsl_t &mutex) : mutexp(&mutex) 
  { 
    //printf("[%d] MUTEX LOCK IN %p\n", gasnet_mynode(), mutexp);
    gasnet_hsl_lock(mutexp); 
    //printf("[%d] MUTEX LOCK HELD %p\n", gasnet_mynode(), mutexp);
  }
  AutoHSLLock(gasnet_hsl_t *_mutexp) : mutexp(_mutexp) 
  { 
    //printf("[%d] MUTEX LOCK IN %p\n", gasnet_mynode(), mutexp);
    gasnet_hsl_lock(mutexp); 
    //printf("[%d] MUTEX LOCK HELD %p\n", gasnet_mynode(), mutexp);
  }
  ~AutoHSLLock(void) 
  { 
    gasnet_hsl_unlock(mutexp);
    //printf("[%d] MUTEX LOCK OUT %p\n", gasnet_mynode(), mutexp);
  }
protected:
  gasnet_hsl_t *mutexp;
};

namespace RegionRuntime {
  namespace LowLevel {

    // for each of the ID-based runtime objects, we're going to have an
    //  implementation class and a table to look them up in
    struct Node {
      Node(void)
      {
	gasnet_hsl_init(&mutex);
	events.reserve(1000);
	locks.reserve(1000);
      }

      gasnet_hsl_t mutex;  // used to cover resizing activities on vectors below
      std::vector<Event::Impl> events;
      std::vector<Lock::Impl> locks;
      std::vector<Memory::Impl *> memories;
      std::vector<Processor::Impl *> processors;
      std::vector<RegionMetaDataUntyped::Impl *> metadatas;
    };

    class ID {
    public:
      // two forms of bit pack for IDs:
      //
      //  3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1
      //  1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
      // +-----+---------------------------------------------------------+
      // | TYP |   NODE  |           INDEX                               |
      // | TYP |   NODE  |   INDEX_H     |           INDEX_L             |
      // +-----+---------------------------------------------------------+

      enum {
	TYPE_BITS = 3,
	INDEX_H_BITS = 8,
	INDEX_L_BITS = 16,
	INDEX_BITS = INDEX_H_BITS + INDEX_L_BITS,
	NODE_BITS = 32 - TYPE_BITS - INDEX_BITS
      };

      enum ID_Types {
	ID_SPECIAL,
	ID_EVENT,
	ID_LOCK,
	ID_MEMORY,
	ID_PROCESSOR,
	ID_METADATA,
	ID_ALLOCATOR,
	ID_INSTANCE,
      };

      enum ID_Specials {
	ID_INVALID = 0,
	ID_GLOBAL_MEM = (1U << INDEX_H_BITS) - 1,
      };

      ID(unsigned _value) : value(_value) {}

      template <class T>
      ID(T thing_to_get_id_from) : value(thing_to_get_id_from.id) {}

      ID(ID_Types _type, unsigned _node, unsigned _index)
	: value((((unsigned)_type) << (NODE_BITS + INDEX_BITS)) |
		(_node << INDEX_BITS) |
		_index) {}

      ID(ID_Types _type, unsigned _node, unsigned _index_h, unsigned _index_l)
	: value((((unsigned)_type) << (NODE_BITS + INDEX_BITS)) |
		(_node << INDEX_BITS) |
		(_index_h << INDEX_L_BITS) |
		_index_l) {}

      unsigned id(void) const { return value; }
      ID_Types type(void) const { return (ID_Types)(value >> (NODE_BITS + INDEX_BITS)); }
      unsigned node(void) const { return ((value >> INDEX_BITS) & ((1U << NODE_BITS)-1)); }
      unsigned index(void) const { return (value & ((1U << INDEX_BITS) - 1)); }
      unsigned index_h(void) const { return ((value >> INDEX_L_BITS) & ((1U << INDEX_H_BITS)-1)); }
      unsigned index_l(void) const { return (value & ((1U << INDEX_L_BITS) - 1)); }

      template <class T>
      T convert(void) const { T thing_to_return = { value }; return thing_to_return; }
      
    protected:
      unsigned value;
    };
    
    class Runtime {
    public:
      static Runtime *get_runtime(void) { return runtime; }

      Event::Impl *get_event_impl(ID id);
      Lock::Impl *get_lock_impl(ID id);
      Memory::Impl *get_memory_impl(ID id);
      Processor::Impl *get_processor_impl(ID id);
      RegionMetaDataUntyped::Impl *get_metadata_impl(ID id);
      RegionAllocatorUntyped::Impl *get_allocator_impl(ID id);
      RegionInstanceUntyped::Impl *get_instance_impl(ID id);

    protected:
    public:
      static Runtime *runtime;

      Node *nodes;
      Memory::Impl *global_memory;
    };

    enum ActiveMessageIDs {
      FIRST_AVAILABLE = 128,
      NODE_ANNOUNCE_MSGID,
      SPAWN_TASK_MSGID,
      LOCK_REQUEST_MSGID,
      LOCK_RELEASE_MSGID,
      LOCK_GRANT_MSGID,
      EVENT_SUBSCRIBE_MSGID,
      EVENT_TRIGGER_MSGID,
      REMOTE_MALLOC_MSGID,
      REMOTE_MALLOC_RPLID,
    };

    /*static*/ Runtime *Runtime::runtime = 0;

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

    class Lock::Impl {
    public:
      Impl(void);

      void init(Lock _me, unsigned _init_owner);

      template <class T>
      void set_local_data(T *data)
      {
	local_data = data;
	local_data_size = sizeof(T);
      }

      //protected:
      Lock me;
      unsigned owner; // which node owns the lock
      unsigned count; // number of locks held by local threads
      unsigned mode;  // lock mode
      bool in_use;

      enum { MODE_EXCL = 0 };

      gasnet_hsl_t *mutex; // controls which local thread has access to internal data (not runtime-visible lock)

      // bitmasks of which remote nodes are waiting on a lock (or sharing it)
      uint64_t remote_waiter_mask, remote_sharer_mask;
      //std::list<LockWaiter *> local_waiters; // set of local threads that are waiting on lock
      std::map<unsigned, std::deque<Event> > local_waiters;
      bool requested; // do we have a request for the lock in flight?

      // local data protected by lock
      void *local_data;
      size_t local_data_size;

      Event lock(unsigned new_mode, bool exclusive,
		 Event after_lock = Event::NO_EVENT);

      bool select_local_waiters(std::deque<Event>& to_wake);

      void unlock(void);
    };

    struct LockRequestArgs {
      gasnet_node_t node;
      Lock lock;
      unsigned mode;
    };

    void handle_lock_request(LockRequestArgs args);

    typedef ActiveMessageShortNoReply<LOCK_REQUEST_MSGID, 
				      LockRequestArgs, 
				      handle_lock_request> LockRequestMessage;

    struct LockReleaseArgs {
      gasnet_node_t node;
      Lock lock;
    };
    
    void handle_lock_release(LockReleaseArgs args);

    typedef ActiveMessageShortNoReply<LOCK_RELEASE_MSGID,
				      LockReleaseArgs,
				      handle_lock_release> LockReleaseMessage;

    struct LockGrantArgs {
      Lock lock;
      unsigned mode;
      uint64_t remote_waiter_mask;
    };

    void handle_lock_grant(LockGrantArgs args, const void *data, size_t datalen);

    typedef ActiveMessageMediumNoReply<LOCK_GRANT_MSGID,
				       LockGrantArgs,
				       handle_lock_grant> LockGrantMessage;

    class RegionMetaDataUntyped::Impl {
    public:
      Impl(RegionMetaDataUntyped _me, size_t _num_elmts, size_t _elmt_size)
	: me(_me)
      {
	data.num_elmts = _num_elmts;
	data.elmt_size = _elmt_size;
	data.master_allocator = RegionAllocatorUntyped::NO_ALLOC;
	data.master_instance = RegionInstanceUntyped::NO_INST;
	lock.init(ID(me).convert<Lock>(), ID(me).node());
	lock.set_local_data(&data);
      }

      ~Impl(void) {}

      RegionAllocatorUntyped get_master_allocator_untyped(void) const
      {
	return data.master_allocator;
      }

      RegionInstanceUntyped get_master_instance_untyped(void) const
      {
	return data.master_instance;
      }

      RegionMetaDataUntyped me;
      Lock::Impl lock;

      struct CoherentData {
	size_t num_elmts, elmt_size;
	RegionAllocatorUntyped master_allocator;
	RegionInstanceUntyped master_instance;
      };

      CoherentData data;
    };

    class RegionAllocatorUntyped::Impl {
    public:
      Impl(RegionAllocatorUntyped _me, RegionMetaDataUntyped _region, Memory _memory, int _mask_start);

      ~Impl(void);

      unsigned alloc_elements(unsigned count = 1);

      void free_elements(unsigned ptr, unsigned count = 1);

      struct CoherentData {
	RegionMetaDataUntyped region;
	Memory memory;
	int mask_start;
      };

      RegionAllocatorUntyped me;
      CoherentData data;

    protected:
      ElementMask avail_elmts, changed_elmts;
    };

    class RegionInstanceUntyped::Impl {
    public:
      Impl(RegionInstanceUntyped _me, RegionMetaDataUntyped _region, Memory _memory, int _offset)
	: me(_me), region(_region), memory(_memory), offset(_offset) {}

      ~Impl(void) {}

      void get_bytes(unsigned ptr_value, void *dst, size_t size);
      void put_bytes(unsigned ptr_value, const void *src, size_t size);

    public: //protected:
      friend class RegionInstanceUntyped;

      RegionInstanceUntyped me;
      RegionMetaDataUntyped region;
      Memory memory;
      int offset;
    };

    ///////////////////////////////////////////////////
    // Events

    class Event::Impl {
    public:
#if 0
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
#endif
      Impl(void)
      {
	Event bad = { -1, -1 };
	init(bad, -1); 
      }

      void init(Event _me, unsigned _init_owner)
      {
	me = _me;
	owner = _init_owner;
	generation = 0;
	gen_subscribed = 0;
	in_use = false;
	mutex = new gasnet_hsl_t;
	//printf("[%d] MUTEX INIT %p\n", gasnet_mynode(), mutex);
	gasnet_hsl_init(mutex);
	remote_waiters = 0;
      }

      static Event create_event(void);

      // test whether an event has triggered without waiting
      bool has_triggered(Event::gen_t needed_gen);

      // causes calling thread to block until event has occurred
      void wait(Event::gen_t needed_gen);

      // creates an event that won't trigger until all input events have
      Event merge_events(const std::set<Event>& wait_for);

      // record that the event has triggered and notify anybody who cares
      void trigger(Event::gen_t gen_triggered, bool local_trigger);

      class EventWaiter {
      public:
	virtual void event_triggered(void) = 0;
      };

      void add_waiter(Event event, EventWaiter *waiter);

    public: //protected:
      Event me;
      unsigned owner;
      Event::gen_t generation, gen_subscribed;
      bool in_use;

      gasnet_hsl_t *mutex; // controls which local thread has access to internal data (not runtime-visible event)

      uint64_t remote_waiters; // bitmask of which remote nodes are waiting on the event
      std::map<Event::gen_t, std::vector<EventWaiter *> > local_waiters; // set of local threads that are waiting on event (keyed by generation)
    };

    struct EventSubscribeArgs {
      gasnet_node_t node;
      Event event;
    };

    void handle_event_subscribe(EventSubscribeArgs args);

    typedef ActiveMessageShortNoReply<EVENT_SUBSCRIBE_MSGID,
				      EventSubscribeArgs,
				      handle_event_subscribe> EventSubscribeMessage;

    void handle_event_trigger(Event event);

    typedef ActiveMessageShortNoReply<EVENT_TRIGGER_MSGID,
				      Event,
				      handle_event_trigger> EventTriggerMessage;

    void handle_event_subscribe(EventSubscribeArgs args)
    {
      printf("[%d] event subscription: node=%d event=%x/%d\n",
	     gasnet_mynode(), args.node, args.event.id, args.event.gen);

      Event::Impl *impl = args.event.impl();

      // early-out case: if we can see the generation needed has already
      //  triggered, signal without taking the mutex
      unsigned stale_gen = impl->generation;
      if(stale_gen >= args.event.gen) {
	printf("[%d] event subscription early-out: node=%d event=%x/%d (<= %d)\n",
	       gasnet_mynode(), args.node, args.event.id, args.event.gen, stale_gen);
	Event e = args.event;
	e.gen = stale_gen;
	EventTriggerMessage::request(args.node, e);
	return;
      }

      {
	AutoHSLLock a(impl->mutex);

	// now that we have the lock, check the needed generation again
	if(impl->generation >= args.event.gen) {
	  printf("[%d] event subscription already done: node=%d event=%x/%d (<= %d)\n",
		 gasnet_mynode(), args.node, args.event.id, args.event.gen, impl->generation);
	  Event e = args.event;
	  e.gen = impl->generation;
	  EventTriggerMessage::request(args.node, e);
	} else {
	  // nope - needed generation hasn't happened yet, so add this node to
	  //  the mask
	  printf("[%d] event subscription recorded: node=%d event=%x/%d (> %d)\n",
		 gasnet_mynode(), args.node, args.event.id, args.event.gen, impl->generation);
	  impl->remote_waiters |= (1ULL << args.node);
	}
      }
    }

    void handle_event_trigger(Event event)
    {
      printf("Remote trigger of event %x/%d!\n", event.id, event.gen);
      event.impl()->trigger(event.gen, false);
    }

    /*static*/ const Event Event::NO_EVENT = Event();

    Event::Impl *Event::impl(void) const
    {
      return Runtime::runtime->get_event_impl(*this);
    }

    bool Event::has_triggered(void) const
    {
      if(!id) return true; // special case: NO_EVENT has always triggered
      Event::Impl *e = Runtime::get_runtime()->get_event_impl(*this);
      return e->has_triggered(gen);
    }

    void Event::wait(void) const
    {
      if(!id) return;  // special case: never wait for NO_EVENT
      Event::Impl *e = Runtime::get_runtime()->get_event_impl(*this);

      void *thread = gasnett_threadkey_get(cur_thread);
      printf("event thread = %p\n", thread);
      e->wait(gen);
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
      impl()->trigger(gen, true);
      //Runtime::get_runtime()->get_event_impl(*this)->trigger();
    }

#if 0
    void trigger_event_handler(Event e)
    {
      Runtime::get_runtime()->get_event_impl(e)->trigger();
    }
#endif

    /*static*/ Event Event::Impl::create_event(void)
    {
      // TODO: figure out if it's safe to iterate over a vector that is
      //  being resized?
      AutoHSLLock a(Runtime::runtime->nodes[gasnet_mynode()].mutex);

      std::vector<Event::Impl>& events = Runtime::runtime->nodes[gasnet_mynode()].events;

      // try to find an event we can reuse
      for(std::vector<Event::Impl>::iterator it = events.begin();
	  it != events.end();
	  it++) {
	// check the owner and in_use without taking the lock - conservative check
	if((*it).in_use || ((*it).owner != gasnet_mynode())) continue;

	// now take the lock and make sure it really isn't in use
	AutoHSLLock a((*it).mutex);
	if(!(*it).in_use && ((*it).owner == gasnet_mynode())) {
	  // now we really have the event
	  (*it).in_use = true;
	  Event ev = (*it).me;
	  ev.gen = (*it).generation + 1;
	  //printf("REUSE EVENT %x/%d\n", ev.id, ev.gen);
	  return ev;
	}
      }

      // couldn't reuse an event - make a new one
      // TODO: take a lock here!?
      unsigned index = events.size();
      events.resize(index + 1);
      Event ev = ID(ID::ID_EVENT, gasnet_mynode(), index).convert<Event>();
      events[index].init(ev, gasnet_mynode());
      events[index].in_use = true;
      ev.gen = 1; // waiting for first generation of this new event
      //printf("NEW EVENT %x/%d\n", ev.id, ev.gen);
      return ev;
    }

    void Event::Impl::add_waiter(Event event, EventWaiter *waiter)
    {
      bool trigger_now = false;

      int subscribe_owner = -1;
      EventSubscribeArgs args;

      {
	AutoHSLLock a(mutex);

	if(event.gen > generation) {
	  printf("[%d] event not ready: event=%x/%d owner=%d gen=%d subscr=%d\n",
		 gasnet_mynode(), event.id, event.gen, owner, generation, gen_subscribed);
	  // we haven't triggered the needed generation yet - add to list of
	  //  waiters, and subscribe if we're not the owner
	  local_waiters[event.gen].push_back(waiter);
	  //printf("LOCAL WAITERS CHECK: %zd\n", local_waiters.size());

	  if((owner != gasnet_mynode()) && (event.gen > gen_subscribed)) {
	    args.node = gasnet_mynode();
	    args.event = event;
	    subscribe_owner = owner;
	    gen_subscribed = event.gen;
	  }
	} else {
	  // event we are interested in has already triggered!
	  trigger_now = true; // actually do trigger outside of mutex
	}
      }

      if(subscribe_owner != -1)
	EventSubscribeMessage::request(owner, args);

      if(trigger_now)
	waiter->event_triggered();
    }

    bool Event::Impl::has_triggered(Event::gen_t needed_gen)
    {
      return (needed_gen < generation);
    }

    void Event::Impl::wait(Event::gen_t needed_gen)
    {
      assert(0);
    }
    
    void Event::Impl::trigger(Event::gen_t gen_triggered, bool local_trigger)
    {
      //printf("[%d] TRIGGER %x/%d\n", gasnet_mynode(), me.id, gen_triggered);
      std::deque<EventWaiter *> to_wake;
      {
	//printf("[%d] TRIGGER MUTEX IN %x/%d\n", gasnet_mynode(), me.id, gen_triggered);
	AutoHSLLock a(mutex);
	//printf("[%d] TRIGGER MUTEX HOLD %x/%d\n", gasnet_mynode(), me.id, gen_triggered);

	//printf("[%d] TRIGGER GEN: %x/%d->%d\n", gasnet_mynode(), me.id, generation, gen_triggered);
	assert((generation + 1) == gen_triggered);
	assert(gen_triggered > generation);

	//printf("[%d] LOCAL WAITERS: %zd\n", gasnet_mynode(), local_waiters.size());
	std::map<Event::gen_t, std::vector<EventWaiter *> >::iterator it = local_waiters.begin();
	while((it != local_waiters.end()) && (it->first <= gen_triggered)) {
	  //printf("[%d] LOCAL WAIT: %d (%zd)\n", gasnet_mynode(), it->first, it->second.size());
	  to_wake.insert(to_wake.end(), it->second.begin(), it->second.end());
	  local_waiters.erase(it);
	  it = local_waiters.begin();
	}

	// notify remote waiters and/or event's actual owner
	if(owner == gasnet_mynode()) {
	  // send notifications to every other node that has subscribed
	  Event ev = me;
	  ev.gen = gen_triggered;
	  for(int node = 0; remote_waiters != 0; node++, remote_waiters >>= 1)
	    if(remote_waiters & 1)
	      EventTriggerMessage::request(node, ev);
	} else {
	  if(local_trigger) {
	    // if we're not the owner, we just send to the owner and let him
	    //  do the broadcast (assuming the trigger was local)
	    assert(remote_waiters == 0);

	    Event ev = me;
	    ev.gen = gen_triggered;
	    EventTriggerMessage::request(owner, ev);
	  }
	}

	generation = gen_triggered;
	in_use = false;
      }

      // now that we've let go of the lock, notify all the waiters who wanted
      //  this event generation (or an older one)
      for(std::deque<EventWaiter *>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++)
	(*it)->event_triggered();
    }

    ///////////////////////////////////////////////////
    // Locks

    /*static*/ const Lock Lock::NO_LOCK = { 0 };

    Lock::Impl *Lock::impl(void) const
    {
      return Runtime::runtime->get_lock_impl(*this);
    }

    Lock::Impl::Impl(void)
    {
      init(Lock::NO_LOCK, -1);
    }

    void Lock::Impl::init(Lock _me, unsigned _init_owner)
    {
      me = _me;
      owner = _init_owner;
      count = 0;
      mode = 0;
      in_use = false;
      mutex = new gasnet_hsl_t;
      gasnet_hsl_init(mutex);
      remote_waiter_mask = 0;
      remote_sharer_mask = 0;
      requested = false;
      local_data = 0;
      local_data_size = 0;
    }

    /*static*/ void /*Lock::Impl::*/handle_lock_request(LockRequestArgs args)
    {
      Lock::Impl *impl = args.lock.impl();

      printf("[%d] lock request: lock=%x, node=%d, mode=%d\n",
	     gasnet_mynode(), args.lock.id, args.node, args.mode);
      {
	AutoHSLLock a(impl->mutex);

	// case 1: we don't even own the lock any more - pass the request on
	//  to whoever we think the owner is
	if(impl->owner != gasnet_mynode()) {
	  // can reuse the args we were given
	  LockRequestMessage::request(impl->owner, args);
	  return;
	}

	// case 2: we're the owner, and nobody is holding the lock, so grant
	//  it to the (original) requestor
	if((impl->count == 0) && (impl->remote_sharer_mask == 0)) {
	  assert(impl->remote_waiter_mask == 0);

	  printf("[%d] granting lock request: lock=%x, node=%d, mode=%d\n",
		 gasnet_mynode(), args.lock.id, args.node, args.mode);
	  LockGrantArgs g_args;
	  g_args.lock = args.lock;
	  g_args.mode = 0; // always give it exclusively for now
	  g_args.remote_waiter_mask = impl->remote_waiter_mask;
	  LockGrantMessage::request(args.node, g_args,
				    impl->local_data, impl->local_data_size);

	  impl->owner = args.node;
	  return;
	}

	// case 3: we're the owner, but we can't grant the lock right now -
	//  just set a bit saying that the node is waiting and get back to
	//  work
	printf("[%d] deferring lock request: lock=%x, node=%d, mode=%d (count=%d cmode=%d)\n",
	       gasnet_mynode(), args.lock.id, args.node, args.mode, impl->count, impl->mode);
	impl->remote_waiter_mask |= (1ULL << args.node);
      }
    }

    /*static*/ void /*Lock::Impl::*/handle_lock_release(LockReleaseArgs args)
    {
      assert(0);
    }

    void handle_lock_grant(LockGrantArgs args, const void *data, size_t datalen)
    {
      printf("[%d] lock request granted: lock=%x mode=%d mask=%llx\n",
	     gasnet_mynode(), args.lock.id, args.mode, args.remote_waiter_mask);

      std::deque<Event> to_wake;

      Lock::Impl *impl = args.lock.impl();
      {
	AutoHSLLock a(impl->mutex);

	// make sure we were really waiting for this lock
	assert(impl->owner != gasnet_mynode());
	assert(impl->requested);

	// first, update our copy of the protected data (if any)
	assert(impl->local_data_size == datalen);
	if(datalen)
	  memcpy(impl->local_data, data, datalen);

	if(args.mode == 0) // take ownership if given exclusive access
	  impl->owner = gasnet_mynode();
	impl->mode = args.mode;
	impl->remote_waiter_mask = args.remote_waiter_mask;
	impl->requested = false;

	bool any_local = impl->select_local_waiters(to_wake);
	assert(any_local);
      }

      for(std::deque<Event>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	printf("[%d] unlock trigger: lock=%x event=%x/%d\n",
	       gasnet_mynode(), args.lock, (*it).id, (*it).gen);
	(*it).impl()->trigger((*it).gen, true);
      }
    }

    Event Lock::Impl::lock(unsigned new_mode, bool exclusive,
			   Event after_lock /*= Event::NO_EVENT*/)
    {
      printf("[%d] local lock request: lock=%x mode=%d excl=%d event=%x/%d\n",
	     gasnet_mynode(), me.id, new_mode, exclusive, after_lock.id, after_lock.gen);

      // collapse exclusivity into mode
      if(exclusive) new_mode = MODE_EXCL;

      AutoHSLLock a(mutex); // hold mutex on lock for entire function

      bool got_lock = false;

      if(owner == gasnet_mynode()) {
	// case 1: we own the lock
	// can we grant it?
	if((count == 0) || ((mode == new_mode) && (mode != MODE_EXCL))) {
	  mode = new_mode;
	  count++;
	  got_lock = true;
	}
      } else {
	// somebody else owns it
	
	// are we sharing?
	if((count > 0) && (mode == new_mode)) {
	  // we're allowed to grant additional sharers with the same mode
	  assert(mode != MODE_EXCL);
	  if(mode == new_mode) {
	    count++;
	    got_lock = true;
	  }
	}
	
	// if we didn't get the lock, we'll have to ask for it from the
	//  other node (even if we're currently sharing with the wrong mode)
	if(!got_lock && !requested) {
	  printf("[%d] requesting lock: lock=%x node=%d mode=%d\n",
		 gasnet_mynode(), me.id, owner, new_mode);
	  LockRequestArgs args;
	  args.node = gasnet_mynode();
	  args.lock = me;
	  args.mode = new_mode;
	  LockRequestMessage::request(owner, args);
	  
	  requested = true;
	}
      }

      // if we got the lock, trigger an event if we're given one
      if(got_lock) {
	if(after_lock.exists()) after_lock.impl()->trigger(after_lock.gen, true);
	return after_lock;
      }

      // if we didn't get the lock, put our event on the queue of local
      //  waiters - create an event if we weren't given one to use
      if(!after_lock.exists())
	after_lock = Event::Impl::create_event();
      local_waiters[mode].push_back(after_lock);
      return after_lock;
    }

    // factored-out code to select one or more local waiters on a lock
    //  fills events to trigger into 'to_wake' and returns true if any were
    //  found - NOTE: ASSUMES LOCK IS ALREADY HELD!
    bool Lock::Impl::select_local_waiters(std::deque<Event>& to_wake)
    {
      if(local_waiters.size() == 0)
	return false;

      // favor the local waiters
      printf("[%d] lock going to local waiter: size=%zd first=%d(%zd)\n",
	     gasnet_mynode(), local_waiters.size(), 
	     local_waiters.begin()->first,
	     local_waiters.begin()->second.size());
	
      // further favor exclusive waiters
      if(local_waiters.find(MODE_EXCL) != local_waiters.end()) {
	std::deque<Event>& excl_waiters = local_waiters[MODE_EXCL];
	to_wake.push_back(excl_waiters.front());
	excl_waiters.pop_front();
	  
	// if the set of exclusive waiters is empty, delete it
	if(excl_waiters.size() == 0)
	  local_waiters.erase(MODE_EXCL);
	  
	mode = MODE_EXCL;
	count = 1;
      } else {
	// pull a whole list of waiters that want to share with the same mode
	std::map<unsigned, std::deque<Event> >::iterator it = local_waiters.begin();
	
	mode = it->first;
	count = it->second.size();

	// grab the list of events wanting to share the lock
	to_wake.swap(it->second);
	  
	// TODO: can we share with any other nodes?
      }

      return true;
    }

    void Lock::Impl::unlock(void)
    {
      // make a list of events that we be woken - can't do it while holding the
      //  lock's mutex (because the event we trigger might try to take the lock)
      std::deque<Event> to_wake;
      {
	printf("[%d] unlock: lock=%x count=%d mode=%d share=%llx wait=%llx\n",
	       gasnet_mynode(), me.id, count, mode, remote_sharer_mask, remote_waiter_mask);
	AutoHSLLock a(mutex); // hold mutex on lock for entire function

	assert(count > 0);

	// if this isn't the last holder of the lock, just decrement count
	//  and return
	count--;
	if(count > 0) return;

	// case 1: if we were sharing somebody else's lock, tell them we're
	//  done
	if(owner != gasnet_mynode()) {
	  assert(mode != MODE_EXCL);
	  mode = 0;
	  LockReleaseArgs args;
	  args.node = gasnet_mynode();
	  args.lock = me;
	  LockReleaseMessage::request(owner, args);
	  return;
	}

	// case 2: we own the lock, so we can give it to another waiter
	//  (local or remote)
	bool any_local = select_local_waiters(to_wake);

	if(!any_local && (remote_waiter_mask != 0)) {
	  // nobody local wants it, but another node does
	  int new_owner = 0;
	  while(((remote_waiter_mask >> new_owner) & 1) == 0) new_owner++;

	  printf("[%d] lock going to remote waiter: new=%d mask=%llx\n",
		 gasnet_mynode(), new_owner, remote_waiter_mask);

	  LockGrantArgs args;
	  args.lock = me;
	  args.mode = 0; // TODO: figure out shared cases
	  args.remote_waiter_mask = remote_waiter_mask & ~(1ULL << new_owner);
	  LockGrantMessage::request(new_owner, args,
				    local_data, local_data_size);

	  owner = new_owner;
	  remote_waiter_mask = 0;
	}
      }

      for(std::deque<Event>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	printf("[%d] unlock trigger: lock=%x event=%x/%d\n",
	       gasnet_mynode(), me.id, (*it).id, (*it).gen);
	(*it).impl()->trigger((*it).gen, true);
      }
    }

    class DeferredLockRequest : public Event::Impl::EventWaiter {
    public:
      DeferredLockRequest(Lock _lock, unsigned _mode, bool _exclusive,
			  Event _after_lock)
	: lock(_lock), mode(_mode), exclusive(_exclusive), after_lock(_after_lock) {}

      virtual void event_triggered(void)
      {
	lock.impl()->lock(mode, exclusive, after_lock);
      }

    protected:
      Lock lock;
      unsigned mode;
      bool exclusive;
      Event after_lock;
    };

    Event Lock::lock(unsigned mode /* = 0 */, bool exclusive /* = true */,
		     Event wait_on /* = Event::NO_EVENT */) const
    {
      printf("LOCK(%x, %d, %d, %x) -> ", id, mode, exclusive, wait_on.id);
      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(wait_on.has_triggered()) {
	Event e = impl()->lock(mode, exclusive);
	printf("(%x/%d)\n", e.id, e.gen);
	return e;
      } else {
	Event after_lock = Event::Impl::create_event();
	wait_on.impl()->add_waiter(wait_on, new DeferredLockRequest(*this, mode, exclusive, after_lock));
	printf("*(%x/%d)\n", after_lock.id, after_lock.gen);
	return after_lock;
      }
    }

    class DeferredUnlockRequest : public Event::Impl::EventWaiter {
    public:
      DeferredUnlockRequest(Lock _lock)
	: lock(_lock) {}

      virtual void event_triggered(void)
      {
	lock.impl()->unlock();
      }

    protected:
      Lock lock;
    };

    // releases a held lock - release can be deferred until an event triggers
    void Lock::unlock(Event wait_on /* = Event::NO_EVENT */) const
    {
      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(wait_on.has_triggered()) {
	impl()->unlock();
      } else {
	wait_on.impl()->add_waiter(wait_on, new DeferredUnlockRequest(*this));
      }
    }

    // Create a new lock, destroy an existing lock
    /*static*/ Lock Lock::create_lock(void)
    {
      // TODO: figure out if it's safe to iterate over a vector that is
      //  being resized?
      AutoHSLLock a(Runtime::runtime->nodes[gasnet_mynode()].mutex);

      std::vector<Lock::Impl>& locks = Runtime::runtime->nodes[gasnet_mynode()].locks;

      // try to find an lock we can reuse
      for(std::vector<Lock::Impl>::iterator it = locks.begin();
	  it != locks.end();
	  it++) {
	// check the owner and in_use without taking the lock - conservative check
	if((*it).in_use || ((*it).owner != gasnet_mynode())) continue;

	// now take the lock and make sure it really isn't in use
	AutoHSLLock a((*it).mutex);
	if(!(*it).in_use && ((*it).owner == gasnet_mynode())) {
	  // now we really have the lock
	  (*it).in_use = true;
	  Lock l = (*it).me;
	  return l;
	}
      }

      // couldn't reuse an lock - make a new one
      // TODO: take a lock here!?
      unsigned index = locks.size();
      locks.resize(index + 1);
      Lock l = ID(ID::ID_LOCK, gasnet_mynode(), index).convert<Lock>();
      locks[index].init(l, gasnet_mynode());
      return l;
    }

    void Lock::destroy_lock()
    {
    }

    ///////////////////////////////////////////////////
    // Memory

    Memory::Impl *Memory::impl(void) const
    {
      return Runtime::runtime->get_memory_impl(*this);
    }

    /*static*/ const Memory Memory::NO_MEMORY = { 0 };

    class Memory::Impl {
    public:
      Impl(Memory _me, size_t _size)
	: me(_me), size(_size)
      {
	gasnet_hsl_init(&mutex);
      }

      unsigned add_allocator(RegionAllocatorUntyped::Impl *a);
      unsigned add_instance(RegionInstanceUntyped::Impl *i);

      RegionAllocatorUntyped::Impl *get_allocator(RegionAllocatorUntyped a);
      RegionInstanceUntyped::Impl *get_instance(RegionInstanceUntyped i);

      RegionAllocatorUntyped create_allocator(RegionMetaDataUntyped r);
      RegionInstanceUntyped create_instance(RegionMetaDataUntyped r);

      void destroy_allocator(RegionAllocatorUntyped a);
      void destroy_instance(RegionInstanceUntyped i);

      virtual int alloc_bytes(size_t size) = 0;
      virtual void free_bytes(int offset, size_t size) = 0;

      virtual void get_bytes(unsigned offset, void *dst, size_t size) = 0;
      virtual void put_bytes(unsigned offset, const void *src, size_t size) = 0;

      virtual void *get_direct_ptr(unsigned offset, size_t size) = 0;

    public:
      Memory me;
      size_t size;
      gasnet_hsl_t mutex; // protection for resizing vectors
      std::vector<RegionAllocatorUntyped::Impl *> allocators;
      std::vector<RegionInstanceUntyped::Impl *> instances;
    };

    struct RemoteMemAllocArgs {
      Memory memory;
      size_t size;
    };

    int handle_remote_mem_alloc(RemoteMemAllocArgs args)
    {
      printf("[%d] handling remote alloc of size %zd\n", gasnet_mynode(), args.size);
      int result = args.memory.impl()->alloc_bytes(args.size);
      printf("[%d] remote alloc will return %d\n", gasnet_mynode(), result);
      return result;
    }

    typedef ActiveMessageShortReply<REMOTE_MALLOC_MSGID, REMOTE_MALLOC_RPLID,
				    RemoteMemAllocArgs, int,
				    handle_remote_mem_alloc> RemoteMemAllocMessage;

    class LocalCPUMemory : public Memory::Impl {
    public:
      LocalCPUMemory(Memory _me, size_t _size) 
	: Memory::Impl(_me, _size)
      {
	base = new char[_size];
	free_blocks[0] = _size;
      }

      virtual ~LocalCPUMemory(void)
      {
	delete[] base;
      }

      virtual int alloc_bytes(size_t size)
      {
	for(std::map<int, int>::iterator it = free_blocks.begin();
	    it != free_blocks.end();
	    it++) {
	  if(it->second == (int)size) {
	    // perfect match
	    int retval = it->first;
	    free_blocks.erase(it);
	    return retval;
	  }

	  if(it->second > (int)size) {
	    // some left over
	    int leftover = it->second - size;
	    int retval = it->first + leftover;
	    it->second = leftover;
	    return retval;
	  }
	}

	// no blocks large enough - boo hoo
	return -1;
      }

      virtual void free_bytes(int offset, size_t size)
      {
	assert(0);
      }

      virtual void get_bytes(unsigned offset, void *dst, size_t size)
      {
	memcpy(dst, base+offset, size);
      }

      virtual void put_bytes(unsigned offset, const void *src, size_t size)
      {
	memcpy(base+offset, src, size);
      }

      virtual void *get_direct_ptr(unsigned offset, size_t size)
      {
	return (base + offset);
      }

    protected:
      char *base;
      std::map<int, int> free_blocks;
    };

    class RemoteMemory : public Memory::Impl {
    public:
      RemoteMemory(Memory _me, size_t _size)
	: Memory::Impl(_me, _size)
      {
      }

      virtual int alloc_bytes(size_t size)
      {
	// RPC over to owner's node for allocation

	RemoteMemAllocArgs args;
	args.memory = me;
	args.size = size;
	int retval = RemoteMemAllocMessage::request(ID(me).node(), args);
	printf("got: %d\n", retval);
	return retval;
      }

      virtual void free_bytes(int offset, size_t size)
      {
	assert(0);
      }

      virtual void get_bytes(unsigned offset, void *dst, size_t size)
      {
	// can't read/write a remote memory
	assert(0);
      }

      virtual void put_bytes(unsigned offset, const void *src, size_t size)
      {
	// can't read/write a remote memory
	assert(0);
      }

      virtual void *get_direct_ptr(unsigned offset, size_t size)
      {
	return 0;
      }
    };

    class GASNetMemory : public Memory::Impl {
    public:
      GASNetMemory(Memory _me) 
	: Memory::Impl(_me, 0 /* we'll calculate it below */)
      {
	num_nodes = gasnet_nodes();
	seginfos = new gasnet_seginfo_t[num_nodes];
	CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );

	size = seginfos[0].size * num_nodes;
	memory_stride = 1024;

	free_blocks[0] = size;
      }

      virtual ~GASNetMemory(void)
      {
      }

      virtual int alloc_bytes(size_t size)
      {
	if(gasnet_mynode() == 0) {
	  // node 0 performs all allocations
	  for(std::map<int, int>::iterator it = free_blocks.begin();
	      it != free_blocks.end();
	      it++) {
	    if(it->second == (int)size) {
	      // perfect match
	      int retval = it->first;
	      free_blocks.erase(it);
	      return retval;
	    }

	    if(it->second > (int)size) {
	      // some left over
	      int leftover = it->second - size;
	      int retval = it->first + leftover;
	      it->second = leftover;
	      return retval;
	    }
	  }

	  // no blocks large enough - boo hoo
	  return -1;
	} else {
	  RemoteMemAllocArgs args;
	  args.memory = me;
	  args.size = size;
	  int retval = RemoteMemAllocMessage::request(0, args);
	  printf("got: %d\n", retval);
	  return retval;
	}
      }

      virtual void free_bytes(int offset, size_t size)
      {
	assert(0);
      }

      virtual void get_bytes(unsigned offset, void *dst, size_t size)
      {
	char *dst_c = (char *)dst;
	while(size > 0) {
	  int blkid = (offset / memory_stride / num_nodes);
	  int node = (offset / memory_stride) % num_nodes;
	  int blkoffset = offset % memory_stride;
	  size_t chunk_size = memory_stride - blkoffset;
	  if(chunk_size > size) chunk_size = size;
	  gasnet_get(dst_c, node, ((char *)seginfos[node].addr)+(blkid * memory_stride)+blkoffset, chunk_size);
	  offset += chunk_size;
	  dst_c += chunk_size;
	  size -= chunk_size;
	}
      }

      virtual void put_bytes(unsigned offset, const void *src, size_t size)
      {
	char *src_c = (char *)src; // dropping const on purpose...
	while(size > 0) {
	  int blkid = (offset / memory_stride / num_nodes);
	  int node = (offset / memory_stride) % num_nodes;
	  int blkoffset = offset % memory_stride;
	  size_t chunk_size = memory_stride - blkoffset;
	  if(chunk_size > size) chunk_size = size;
	  gasnet_put(node, ((char *)seginfos[node].addr)+(blkid * memory_stride)+blkoffset, src_c, chunk_size);
	  offset += chunk_size;
	  src_c += chunk_size;
	  size -= chunk_size;
	}
      }

      virtual void *get_direct_ptr(unsigned offset, size_t size)
      {
	return 0;  // can't give a pointer to the caller - have to use RDMA
      }

    protected:
      int num_nodes;
      int memory_stride;
      gasnet_seginfo_t *seginfos;
      std::map<int, int> free_blocks;
    };

    RegionAllocatorUntyped Memory::Impl::create_allocator(RegionMetaDataUntyped r)
    {

      // first, we'll need to allocate some storage for the masks
      RegionMetaDataUntyped::Impl *r_impl = r.impl();
      size_t mask_size = ElementMaskImpl::bytes_needed(0, r_impl->data.num_elmts);
      int mask_start = alloc_bytes(2 * mask_size);

      // now find/make an available index to store this in
      unsigned index;
      {
	unsigned size = allocators.size();
	for(index = 0; index < size; index++)
	  if(!allocators[index])
	    break;

	if(index >= size) allocators.push_back(0);
      }

      RegionAllocatorUntyped a = ID(ID::ID_ALLOCATOR, 
				    ID(me).node(),
				    ID(me).index_h(),
				    index).convert<RegionAllocatorUntyped>();
      RegionAllocatorUntyped::Impl *a_impl = new RegionAllocatorUntyped::Impl(a, r, me, mask_start);
      allocators[index] = a_impl;
      return a;
    }

    RegionInstanceUntyped Memory::Impl::create_instance(RegionMetaDataUntyped r)
    {
      // allocator storage for this instance
      RegionMetaDataUntyped::Impl *r_impl = r.impl();
      size_t inst_bytes = r_impl->data.num_elmts * r_impl->data.elmt_size;
      int inst_offset = alloc_bytes(inst_bytes);

      // find/make an available index to store this in
      unsigned index;
      {
	unsigned size = instances.size();
	for(index = 0; index < size; index++)
	  if(!instances[index]) 
	    break;

	if(index >= size) instances.push_back(0);
      }

      RegionInstanceUntyped i = ID(ID::ID_INSTANCE, 
				   ID(me).node(),
				   ID(me).index_h(),
				   index).convert<RegionInstanceUntyped>();

      //RegionMetaDataImpl *r_impl = Runtime::runtime->get_metadata_impl(r);

      RegionInstanceUntyped::Impl *i_impl = new RegionInstanceUntyped::Impl(i, r, me, inst_offset);

      instances[index] = i_impl;

      return i;
    }

    unsigned Memory::Impl::add_allocator(RegionAllocatorUntyped::Impl *a)
    {
      unsigned size = allocators.size();
      for(unsigned index = 0; index < size; index++)
	if(!allocators[index]) {
	  allocators[index] = a;
	  return index;
	}

      allocators.push_back(a);
      return size;
    }

    RegionAllocatorUntyped::Impl *Memory::Impl::get_allocator(RegionAllocatorUntyped a)
    {
      ID id(a);

      // have we heard of this one before?  if not, add it
      unsigned index = id.index_l();
      if(index >= allocators.size()) { // lock not held - just for early out
	AutoHSLLock a(mutex);
	if(index >= allocators.size()) // real check
	  allocators.resize(index + 1);
      }

      if(!allocators[index]) {
	allocators[index] = new RegionAllocatorUntyped::Impl(a, 
							     RegionMetaDataUntyped::NO_REGION,
							     me,
							     -1);
      }

      return allocators[index];
    }

    RegionInstanceUntyped::Impl *Memory::Impl::get_instance(RegionInstanceUntyped i)
    {
      ID id(i);

      // have we heard of this one before?  if not, add it
      unsigned index = id.index_l();
      if(index >= instances.size()) { // lock not held - just for early out
	AutoHSLLock a(mutex);
	if(index >= instances.size()) // real check
	  instances.resize(index + 1);
      }

      if(!instances[index]) {
	//instances[index] = new RegionInstanceImpl(id.node());
	assert(0);
      }

      return instances[index];
    }

    void Memory::Impl::destroy_allocator(RegionAllocatorUntyped i)
    {
      ID id(i);

      // TODO: actually free corresponding storage

      unsigned index = id.index_l();
      assert(index < allocators.size());
      delete allocators[index];
      allocators[index] = 0;
    }

    void Memory::Impl::destroy_instance(RegionInstanceUntyped i)
    {
      ID id(i);

      // TODO: actually free corresponding storage

      unsigned index = id.index_l();
      assert(index < instances.size());
      delete instances[index];
      instances[index] = 0;
    }

    ///////////////////////////////////////////////////
    // Processor

    // global because I'm being lazy...
    static Processor::TaskIDTable task_id_table;

    Processor::Impl *Processor::impl(void) const
    {
      return Runtime::runtime->get_processor_impl(*this);
    }

    class Processor::Impl {
    public:
      Impl(Processor _me, Processor::Kind _kind)
	: me(_me), kind(_kind) {}

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      Event start_event, Event finish_event) = 0;

    public:
      Processor me;
      Processor::Kind kind;
    };

    class LocalProcessor : public Processor::Impl {
    public:
      // simple task object keeps a copy of args
      class Task {
      public:
	Task(LocalProcessor *_proc,
	     Processor::TaskFuncID _func_id,
	     const void *_args, size_t _arglen,
	     Event _finish_event)
	  : proc(_proc), func_id(_func_id), arglen(_arglen), finish_event(_finish_event)
	{
	  if(arglen) {
	    args = malloc(arglen);
	    memcpy(args, _args, arglen);
	  } else {
	    args = 0;
	  }
	}

	virtual ~Task(void)
	{
	  if(args) free(args);
	}

	void run(void)
	{
	  Processor::TaskFuncPtr fptr = task_id_table[func_id];
	  char argstr[100];
	  argstr[0] = 0;
	  for(size_t i = 0; (i < arglen) && (i < 40); i++)
	    sprintf(argstr+2*i, "%02x", ((unsigned *)args)[i]);
	  if(arglen > 40) strcpy(argstr+80, "...");
	  printf("[%d] task start: %d (%p) (%s)\n", gasnet_mynode(), func_id, fptr, argstr);
	  printf("PROC = %p\n", proc);
	  printf("ME = %x\n", proc->me.id);
	  printf("PROC = %p\n", proc);
	  (*fptr)(args, arglen, proc->me);
	  printf("[%d] task end: %d (%p) (%s)\n", gasnet_mynode(), func_id, fptr, argstr);
	  if(finish_event.exists())
	    finish_event.impl()->trigger(finish_event.gen, true);
	}

	LocalProcessor *proc;
	Processor::TaskFuncID func_id;
	void *args;
	size_t arglen;
	Event finish_event;
      };

      // simple thread object just has a task field that you can set and 
      //  wake up to run
      class Thread {
      public:
	enum State { STATE_INIT, STATE_START, STATE_IDLE, STATE_RUN, STATE_SUSPEND };

	Thread(LocalProcessor *_proc) : proc(_proc), task(0), state(STATE_INIT)
	{
	  gasnett_cond_init(&condvar);
	}

	~Thread(void) {}

	void start(void) {
	  state = STATE_START;

	  pthread_attr_t attr;
	  CHECK_PTHREAD( pthread_attr_init(&attr) );
	  CHECK_PTHREAD( pthread_create(&thread, &attr, &thread_main, (void *)this) );
	  CHECK_PTHREAD( pthread_attr_destroy(&attr) );
	}

	void set_task_and_wake(Task *new_task)
	{
	  // assumes proc's mutex already held
	  task = new_task;
	  gasnett_cond_signal(&condvar);
	}

	static void *thread_main(void *args)
	{
	  Thread *me = (Thread *)args;

	  // first thing - take the lock and set our status
	  me->state = STATE_IDLE;

	  // stuff our pointer into TLS so Event::wait can find it
	  gasnett_threadkey_set(cur_thread, me);

	  printf("[%d] worker thread ready: proc=%x\n", gasnet_mynode(), me->proc->me.id);

	  while(1) {
	    // see if there's work sitting around (and we're allowed to 
	    //  start running) - if not, sleep until somebody assigns us work
	    if((me->proc->ready_tasks.size() > 0) &&
	       (me->proc->active_thread_count < me->proc->max_active_threads)) {
	      me->task = me->proc->ready_tasks.front();
	      me->proc->ready_tasks.pop_front();
	      me->proc->active_thread_count++;
	      printf("[%d] thread claiming ready task: proc=%x task=%p thread=%p\n", gasnet_mynode(), me->proc->me.id, me->task, me);
	    } else {
	      me->state = STATE_IDLE;
	      me->proc->avail_threads.push_back(me);
	      printf("[%d] no task for thread, sleeping: proc=%x thread=%p\n",
		     gasnet_mynode(), me->proc->me.id, me);
	      fflush(stdout);
	      gasnett_cond_wait(&me->condvar, &me->proc->mutex.lock);

	      // when we wake up, expect to have a task
	      assert(me->task != 0);
	    }

	    me->state = STATE_RUN;

	    // release lock while task is running
	    gasnet_hsl_unlock(&me->proc->mutex);
	    
	    me->task->run();

	    // TODO: delete task?
	    me->task = 0;

	    // retake lock, decrement active thread count
	    gasnet_hsl_lock(&me->proc->mutex);
	    me->proc->active_thread_count--;
	  }
	}

      public:
	LocalProcessor *proc;
	Task *task;
	State state;
	pthread_t thread;
	gasnett_cond_t condvar;
      };

      class DeferredTaskSpawn : public Event::Impl::EventWaiter {
      public:
	DeferredTaskSpawn(Task *_task) : task(_task) {}

	virtual ~DeferredTaskSpawn(void)
	{
	  // we do _NOT_ own the task - do not free it
	}

	virtual void event_triggered(void)
	{
	  printf("[%d] deferred task now ready: func=%d finish=%x/%d\n",
		 gasnet_mynode(), task->func_id, 
		 task->finish_event.id, task->finish_event.gen);

	  // add task to processor's ready queue
	  task->proc->add_ready_task(task);
	}

      protected:
	Task *task;
      };

      LocalProcessor(Processor _me, int _core_id, 
		     int _total_threads = 1, int _max_active_threads = 1)
	: Processor::Impl(_me, Processor::LOC_PROC), core_id(_core_id),
	  active_thread_count(0), max_active_threads(_max_active_threads)
      {
	// create worker threads - they will enqueue themselves when
	//   they're ready
	for(int i = 0; i < _total_threads; i++) {
	  Thread *t = new Thread(this);
	  printf("[%d] creating worker thread : proc=%x thread=%p\n", gasnet_mynode(), me.id, t);
	  t->start();
	}
      }

      ~LocalProcessor(void)
      {
      }

      void add_ready_task(Task *task)
      {
	// modifications to task/thread lists require mutex
	AutoHSLLock a(mutex);

	// do we have an available thread that can run this task right now?
	if((avail_threads.size() > 0) &&
	   (active_thread_count < max_active_threads)) {
	  Thread *thread = avail_threads.front();
	  avail_threads.pop_front();
	  active_thread_count++;
	  printf("[%d] assigning new task to thread: proc=%x task=%p thread=%p\n", gasnet_mynode(), me.id, task, thread);
	  thread->set_task_and_wake(task);
	} else {
	  // no?  just stuff it on ready list
	  printf("[%d] no thread available for new task: proc=%x task=%p\n", gasnet_mynode(), me.id, task);
	  ready_tasks.push_back(task);
	}
      }

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      Event start_event, Event finish_event)
      {
	// create task object to hold args, etc.
	Task *task = new Task(this, func_id, args, arglen, finish_event);

	// early out - if the event has obviously triggered (or is NO_EVENT)
	//  don't build up continuation
	if(start_event.has_triggered()) {
	  add_ready_task(task);
	} else {
	  printf("[%d] deferring spawn: func=%d event=%x/%d\n",
		 gasnet_mynode(), func_id, start_event.id, start_event.gen);
	  start_event.impl()->add_waiter(start_event, new DeferredTaskSpawn(task));
	}
      }

    protected:
      int core_id;
      int active_thread_count, max_active_threads;
      std::list<Task *> ready_tasks;
      std::list<Thread *> avail_threads;
      gasnet_hsl_t mutex;
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
      Processor::Impl *p = args.proc.impl();
      printf("[%d] remote spawn request: proc_id=%x task_id=%d event=%x/%d\n",
	     gasnet_mynode(), args.proc.id, args.func_id, args.start_event.id, args.start_event.gen);
      p->spawn_task(args.func_id, data, datalen,
		    args.start_event, args.finish_event);
    }

    typedef ActiveMessageMediumNoReply<SPAWN_TASK_MSGID,
				       SpawnTaskArgs,
				       handle_spawn_task_message> SpawnTaskMessage;

    class RemoteProcessor : public Processor::Impl {
    public:
      RemoteProcessor(Processor _me, Processor::Kind _kind)
	: Processor::Impl(_me, _kind)
      {
      }

      ~RemoteProcessor(void)
      {
      }

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      Event start_event, Event finish_event)
      {
	printf("[%d] spawning remote task: proc=%x task=%d start=%x/%d finish=%x/%d\n",
	       gasnet_mynode(), me.id, func_id, 
	       start_event.id, start_event.gen,
	       finish_event.id, finish_event.gen);
	SpawnTaskArgs msgargs;
	msgargs.proc = me;
	msgargs.func_id = func_id;
	msgargs.start_event = start_event;
	msgargs.finish_event = finish_event;
	SpawnTaskMessage::request(ID(me).node(), msgargs, args, arglen);
      }
    };

    Event Processor::spawn(TaskFuncID func_id, const void *args, size_t arglen,
			   Event wait_on) const
    {
      Processor::Impl *p = impl();
      Event finish_event = Event::Impl::create_event();
      p->spawn_task(func_id, args, arglen, wait_on, finish_event);
      return finish_event;
    }

    ///////////////////////////////////////////////////
    // Runtime

    Event::Impl *Runtime::get_event_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_EVENT:
	{
	  Node *n = &runtime->nodes[id.node()];

	  unsigned index = id.index();
	  if(index >= n->events.size()) {
	    AutoHSLLock a(n->mutex); // take lock before we actually resize

	    // grow our array to mirror additions by other nodes
	    //  this should never happen for our own node
	    assert(id.node() != gasnet_mynode());

	    unsigned oldsize = n->events.size();
	    if(index >= oldsize) { // only it's still too small
	      n->events.resize(index + 1);
	      for(unsigned i = oldsize; i <= index; i++)
		n->events[i].init(ID(ID::ID_EVENT, id.node(), i).convert<Event>(),
				  id.node());
	    }
	  }
	  return &(n->events[index]);
	}

      default:
	assert(0);
      }
    }

    Lock::Impl *Runtime::get_lock_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_LOCK:
	{
	  Node *n = &runtime->nodes[id.node()];
	  std::vector<Lock::Impl>& locks = nodes[id.node()].locks;

	  unsigned index = id.index();
	  if(index >= locks.size()) {
	    AutoHSLLock a(n->mutex); // take lock before we actually resize

	    // grow our array to mirror additions by other nodes
	    //  this should never happen for our own node
	    assert(id.node() != gasnet_mynode());

	    unsigned oldsize = n->locks.size();
	    if(index >= oldsize) { // only it's still too small
	      n->locks.resize(index + 1);
	      for(unsigned i = oldsize; i <= index; i++)
		n->locks[i].init(ID(ID::ID_LOCK, id.node(), i).convert<Lock>(),
				 id.node());
	    }
	  }
	  return &(locks[index]);
	}

      case ID::ID_METADATA:
	return &(get_metadata_impl(id)->lock);

      default:
	assert(0);
      }
    }

    template <class T>
    inline T *null_check(T *ptr)
    {
      assert(ptr != 0);
      return ptr;
    }

    Memory::Impl *Runtime::get_memory_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_MEMORY:
      case ID::ID_ALLOCATOR:
      case ID::ID_INSTANCE:
	if(id.index_h() == ID::ID_GLOBAL_MEM)
	  return runtime->global_memory;
	return null_check(runtime->nodes[id.node()].memories[id.index_h()]);

      default:
	assert(0);
      }
    }

    Processor::Impl *Runtime::get_processor_impl(ID id)
    {
      assert(id.type() == ID::ID_PROCESSOR);
      return null_check(runtime->nodes[id.node()].processors[id.index()]);
    }

    RegionMetaDataUntyped::Impl *Runtime::get_metadata_impl(ID id)
    {
      assert(id.type() == ID::ID_METADATA);

      Node *n = &runtime->nodes[id.node()];

      unsigned index = id.index();
      if(index >= n->metadatas.size()) {
	AutoHSLLock a(n->mutex); // take lock before we actually resize

	if(index >= n->metadatas.size())
	  n->metadatas.resize(index + 1);
      }

      if(!n->metadatas[index]) { // haven't seen this metadata before?
	printf("UNKNOWN METADATA %x\n", id.id());
	AutoHSLLock a(n->mutex); // take lock before we actually allocate
	if(!n->metadatas[index]) {
	  n->metadatas[index] = new RegionMetaDataUntyped::Impl(id.convert<RegionMetaDataUntyped>(),
								0, 0);
	} else
	  printf("false alarm...\n");
      }

      return n->metadatas[index];
    }

    RegionAllocatorUntyped::Impl *Runtime::get_allocator_impl(ID id)
    {
      assert(id.type() == ID::ID_ALLOCATOR);
      Memory::Impl *mem = get_memory_impl(id);
      return null_check(mem->allocators[id.index_l()]);
    }

    RegionInstanceUntyped::Impl *Runtime::get_instance_impl(ID id)
    {
      assert(id.type() == ID::ID_INSTANCE);
      Memory::Impl *mem = get_memory_impl(id);
      return null_check(mem->instances[id.index_l()]);
    }

    ///////////////////////////////////////////////////
    // RegionMetaData

    /*static*/ const RegionMetaDataUntyped RegionMetaDataUntyped::NO_REGION = RegionMetaDataUntyped();

    RegionMetaDataUntyped::Impl *RegionMetaDataUntyped::impl(void) const
    {
      return Runtime::runtime->get_metadata_impl(*this);
    }

    /*static*/ RegionMetaDataUntyped RegionMetaDataUntyped::create_region_untyped(Memory memory, size_t num_elmts, size_t elmt_size)
    {
      // find an available ID
      std::vector<RegionMetaDataUntyped::Impl *>& metadatas = Runtime::runtime->nodes[gasnet_mynode()].metadatas;

      unsigned index = 0;
      {
	AutoHSLLock a(Runtime::runtime->nodes[gasnet_mynode()].mutex);

	while((index < metadatas.size()) && (metadatas[index] != 0)) index++;
	if(index >= metadatas.size()) metadatas.resize(index + 1);
	// assign a dummy, but non-zero value to reserve this entry
	// this lets us do the object creation outside the critical section
	metadatas[index] = (RegionMetaDataUntyped::Impl *)1;
      }

      RegionMetaDataUntyped r = ID(ID::ID_METADATA, 
				   gasnet_mynode(), 
				   index).convert<RegionMetaDataUntyped>();

      RegionMetaDataUntyped::Impl *impl = new RegionMetaDataUntyped::Impl(r, num_elmts, elmt_size);
      metadatas[index] = impl;
      
      impl->data.master_allocator = r.create_allocator_untyped(memory);
      impl->data.master_instance = r.create_instance_untyped(memory);

      return r;
    }

    RegionAllocatorUntyped RegionMetaDataUntyped::create_allocator_untyped(Memory memory) const
    {
      ID id(memory);

      Memory::Impl *m_impl = Runtime::runtime->get_memory_impl(memory);

      return m_impl->create_allocator(*this);
#if 0
      RegionAllocatorImpl *a_impl = new RegionAllocatorImpl(id.node());
      unsigned index = m_impl->add_allocator(a_impl);

      ID new_id(ID::ID_ALLOCATOR, id.node(), index);
      RegionAllocatorUntyped r = { new_id.id() };
      return r;
#endif
    }

    RegionInstanceUntyped RegionMetaDataUntyped::create_instance_untyped(Memory memory) const
    {
      
      ID id(memory);

      Memory::Impl *m_impl = Runtime::runtime->get_memory_impl(memory);

      return m_impl->create_instance(*this);
#if 0
      RegionInstanceImpl *i_impl = new RegionInstanceImpl(id.node());
      unsigned index = m_impl->add_instance(i_impl);

      ID new_id(ID::ID_INSTANCE, id.node(), index);
      RegionInstanceUntyped r = { new_id.id() };
      return r;
#endif
    }

    void RegionMetaDataUntyped::destroy_region_untyped(void) const
    {
      assert(0);
    }

    void RegionMetaDataUntyped::destroy_allocator_untyped(RegionAllocatorUntyped allocator) const
    {
      allocator.impl()->data.memory.impl()->destroy_allocator(allocator);
    }

    void RegionMetaDataUntyped::destroy_instance_untyped(RegionInstanceUntyped instance) const
    {
      instance.impl()->memory.impl()->destroy_instance(instance);
    }

    Lock RegionMetaDataUntyped::get_lock(void) const
    {
      // we use our own ID as an ID for our lock
      Lock l = { id };
      return l;
    }

    RegionAllocatorUntyped RegionMetaDataUntyped::get_master_allocator_untyped(void)
    {
      return impl()->get_master_allocator_untyped();
    }

    RegionInstanceUntyped RegionMetaDataUntyped::get_master_instance_untyped(void)
    {
      return impl()->get_master_instance_untyped();
    }

    ///////////////////////////////////////////////////
    // Element Masks

    ElementMask::ElementMask(void)
      : first_element(-1), num_elements(-1), memory(Memory::NO_MEMORY), offset(-1),
	raw_data(0)
    {
    }

    ElementMask::ElementMask(int _num_elements, int _first_element /*= 0*/)
      : first_element(_first_element), num_elements(_num_elements), memory(Memory::NO_MEMORY), offset(-1)
    {
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = calloc(1, bytes_needed);
      //((ElementMaskImpl *)raw_data)->count = num_elements;
      //((ElementMaskImpl *)raw_data)->offset = first_element;
    }

    void ElementMask::init(int _first_element, int _num_elements, Memory _memory, int _offset)
    {
      first_element = _first_element;
      num_elements = _num_elements;
      memory = _memory;
      offset = _offset;
      size_t bytes_needed = ElementMaskImpl::bytes_needed(first_element, num_elements);
      raw_data = memory.impl()->get_direct_ptr(offset, bytes_needed);
    }

    void ElementMask::enable(int start, int count /*= 1*/)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	printf("ENABLE %p %d %d %d %x\n", raw_data, offset, start, count, impl->bits[0]);
	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  unsigned *ptr = &(impl->bits[pos >> 5]);
	  *ptr |= (1U << (pos & 0x1f));
	  pos++;
	}
	printf("ENABLED %p %d %d %d %x\n", raw_data, offset, start, count, impl->bits[0]);
      } else {
	printf("ENABLE(2) %x %d %d %d\n", memory.id, offset, start, count);
	Memory::Impl *m_impl = memory.impl();

	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  unsigned ofs = offset + ((pos >> 5) << 2);
	  unsigned val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  printf("ENABLED(2) %d,  %x\n", ofs, val);
	  val |= (1U << (pos & 0x1f));
	  m_impl->put_bytes(ofs, &val, sizeof(val));
	  pos++;
	}
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
	printf("DISABLE(2) %x %d %d %d\n", memory.id, offset, start, count);
	Memory::Impl *m_impl = memory.impl();

	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  unsigned ofs = offset + ((pos >> 5) << 2);
	  unsigned val;
	  m_impl->get_bytes(ofs, &val, sizeof(val));
	  printf("DISABLED(2) %d,  %x\n", ofs, val);
	  val &= ~(1U << (pos & 0x1f));
	  m_impl->put_bytes(ofs, &val, sizeof(val));
	  pos++;
	}
      }
    }

    int ElementMask::find_enabled(int count /*= 1 */)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	printf("FIND_ENABLED %p %d %d %x\n", raw_data, first_element, count, impl->bits[0]);
	for(int pos = 0; pos <= num_elements - count; pos++) {
	  int run = 0;
	  while(1) {
	    unsigned bit = ((impl->bits[pos >> 5] >> (pos & 0x1f))) & 1;
	    if(bit != 1) break;
	    pos++; run++;
	    if(run >= count) return pos - run;
	  }
	}
      } else {
	Memory::Impl *m_impl = memory.impl();
	printf("FIND_ENABLED(2) %x %d %d %d\n", memory.id, offset, first_element, count);
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

    size_t ElementMask::raw_size(void) const
    {
      return 0;
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

	// if we fall off the end, there's no more ranges to enumerate
	return false;
      } else {
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

	// if we fall off the end, there's no more ranges to enumerate
	return false;
      }
    }

    ///////////////////////////////////////////////////
    // Region Allocators

    /*static*/ const RegionAllocatorUntyped RegionAllocatorUntyped::NO_ALLOC = RegionAllocatorUntyped();

    RegionAllocatorUntyped::Impl *RegionAllocatorUntyped::impl(void) const
    {
      return Runtime::runtime->get_allocator_impl(*this);
    }

    unsigned RegionAllocatorUntyped::alloc_untyped(unsigned count /*= 1*/) const
    {
      return impl()->alloc_elements(count);
    }

    void RegionAllocatorUntyped::free_untyped(unsigned ptr, unsigned count /*= 1  */) const
    {
      return impl()->free_elements(ptr, count);
    }

    RegionAllocatorUntyped::Impl::Impl(RegionAllocatorUntyped _me, RegionMetaDataUntyped _region, Memory _memory, int _mask_start)
      : me(_me)
    {
      data.region = _region;
      data.memory = _memory;
      data.mask_start = _mask_start;

      if(_region.exists()) {
	RegionMetaDataUntyped::Impl *r_impl = _region.impl();
	int num_elmts = r_impl->data.num_elmts;
	int mask_bytes = ElementMaskImpl::bytes_needed(0, num_elmts);
	avail_elmts.init(0, num_elmts, _memory, _mask_start);
	avail_elmts.enable(0, num_elmts);
	changed_elmts.init(0, num_elmts, _memory, _mask_start + mask_bytes);
      }
    }

#if 0
    RegionAllocatorImpl::RegionAllocatorImpl(size_t num_elmts, Memory m)
      : avail_elmts(num_elmts), changed_elmts(num_elmts)
    {
    }
#endif

    RegionAllocatorUntyped::Impl::~Impl(void)
    {
    }

    unsigned RegionAllocatorUntyped::Impl::alloc_elements(unsigned count /*= 1 */)
    {
      int start = avail_elmts.find_enabled(count);
      assert(start >= 0);
      avail_elmts.disable(start, count);
      changed_elmts.enable(start, count);
      return start;
    }

    void RegionAllocatorUntyped::Impl::free_elements(unsigned ptr, unsigned count /*= 1*/)
    {
      avail_elmts.enable(ptr, count);
      changed_elmts.enable(ptr, count);
    }

    ///////////////////////////////////////////////////
    // Region Instances

    RegionInstanceUntyped::Impl *RegionInstanceUntyped::impl(void) const
    {
      return Runtime::runtime->get_instance_impl(*this);
    }

    void RegionInstanceUntyped::Impl::get_bytes(unsigned ptr_value, void *dst, size_t size)
    {
      Memory::Impl *m = Runtime::runtime->get_memory_impl(memory);
      m->get_bytes(offset + ptr_value, dst, size);
    }

    void RegionInstanceUntyped::Impl::put_bytes(unsigned ptr_value, const void *src, size_t size)
    {
      Memory::Impl *m = Runtime::runtime->get_memory_impl(memory);
      m->put_bytes(offset + ptr_value, src, size);
    }

    /*static*/ const RegionInstanceUntyped RegionInstanceUntyped::NO_INST = RegionInstanceUntyped();

    // a generic accessor just holds a pointer to the impl and passes all 
    //  requests through
    RegionInstanceAccessorUntyped<AccessorGeneric> RegionInstanceUntyped::get_accessor_untyped(void) const
    {
      return RegionInstanceAccessorUntyped<AccessorGeneric>((void *)impl());
    }

    Event RegionInstanceUntyped::copy_to_untyped(RegionInstanceUntyped target, 
						 Event wait_on /*= Event::NO_EVENT*/)
    {
      printf("COPY %x -> %x\n", id, target.id);
      RegionInstanceUntyped::Impl *src_impl = impl();
      RegionInstanceUntyped::Impl *dst_impl = target.impl();
      RegionMetaDataUntyped::Impl *r_impl = src_impl->region.impl();
      const void *src_ptr = src_impl->memory.impl()->get_direct_ptr(src_impl->offset, r_impl->data.num_elmts * r_impl->data.elmt_size);
      void *dst_ptr = dst_impl->memory.impl()->get_direct_ptr(dst_impl->offset, r_impl->data.num_elmts * r_impl->data.elmt_size);

      if(src_ptr && dst_ptr) {
	// local copy
	memcpy(dst_ptr, src_ptr, r_impl->data.num_elmts * r_impl->data.elmt_size);
      } else {
	assert(0);
      }
      return Event::NO_EVENT;
    }

    Event RegionInstanceUntyped::copy_to_untyped(RegionInstanceUntyped target,
						 const ElementMask &mask,
						 Event wait_on /*= Event::NO_EVENT*/)
    {
      assert(0);
    }

    void RegionInstanceAccessorUntyped<AccessorGeneric>::get_untyped(unsigned ptr_value, void *dst, size_t size) const
    {
      ((RegionInstanceUntyped::Impl *)internal_data)->get_bytes(ptr_value, dst, size);
    }

    void RegionInstanceAccessorUntyped<AccessorGeneric>::put_untyped(unsigned ptr_value, const void *src, size_t size) const
    {
      ((RegionInstanceUntyped::Impl *)internal_data)->put_bytes(ptr_value, src, size);
    }

    template <>
    bool RegionInstanceAccessorUntyped<AccessorGeneric>::can_convert<AccessorGeneric>(void) const
    { return true; }
    
    template <>
    RegionInstanceAccessorUntyped<AccessorGeneric> RegionInstanceAccessorUntyped<AccessorGeneric>::convert<AccessorGeneric>(void) const
    { return *this; }
    
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
      unsigned num_memories;
    };

    static gasnet_hsl_t announcement_mutex = GASNET_HSL_INITIALIZER;
    static int announcements_received = 0;

    void node_announce_handler(NodeAnnounceData data)
    {
      printf("%d: received announce from %d (%d procs, %d memories)\n", gasnet_mynode(), data.node_id, data.num_procs, data.num_memories);
      Node *n = &(Runtime::get_runtime()->nodes[data.node_id]);
      n->processors.resize(data.num_procs);
      for(unsigned i = 0; i < data.num_procs; i++) {
	Processor p = ID(ID::ID_PROCESSOR, data.node_id, i).convert<Processor>();
	n->processors[i] = new RemoteProcessor(p, Processor::LOC_PROC);
      }

      n->memories.resize(data.num_memories);
      for(unsigned i = 0; i < data.num_memories; i++) {
	Memory m = ID(ID::ID_MEMORY, data.node_id, i).convert<Memory>();
	n->memories[i] = new RemoteMemory(m, 16 << 20);
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
	// see if we've been spawned by gasnet or been run directly
	bool in_gasnet_spawn = false;
	for(int i = 0; i < *argc; i++)
	  if(!strncmp((*argv)[i], "-GASNET", 7))
	    in_gasnet_spawn = true;

	if(!in_gasnet_spawn) {
	  printf("doesn't look like this was called from gasnetrun - lemme try to spawn it for you...\n");
	  int np = 1;
	  const char *p = getenv("GASNET_SSH_SERVERS");
	  if(p)
	    while(*p)
	      if(*p++ == ',') np++;
	  char np_str[10];
	  sprintf(np_str, "%d", np);
	  const char **new_argv = new const char *[*argc + 4];
	  new_argv[0] = "gasnetrun_ibv";
	  new_argv[1] = "-n";
	  new_argv[2] = np_str;
	  for(int i = 0; i < *argc; i++) new_argv[i+3] = (*argv)[i];
	  new_argv[*argc + 3] = 0;
	  execvp(new_argv[0], new_argv);
	  perror("execvp");
	}
	  
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
	
	//gasnet_seginfo_t seginfos = new gasnet_seginfo_t[num_nodes];
	//CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );

	Runtime *r = Runtime::runtime = new Runtime;
	r->nodes = new Node[num_nodes];

	r->global_memory = new GASNetMemory(ID(ID::ID_MEMORY, 0, ID::ID_GLOBAL_MEM, 0).convert<Memory>());

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
      // see if we've been spawned by gasnet or been run directly
      bool in_gasnet_spawn = false;
      for(int i = 0; i < *argc; i++)
	if(!strncmp((*argv)[i], "-GASNET", 7))
	  in_gasnet_spawn = true;

      if(!in_gasnet_spawn) {
	printf("doesn't look like this was called from gasnetrun - lemme try to spawn it for you...\n");
	int np = 1;
	const char *p = getenv("GASNET_SSH_SERVERS");
	if(p)
	  while(*p)
	    if(*p++ == ',') np++;
	char np_str[10];
	sprintf(np_str, "%d", np);
	const char **new_argv = new const char *[*argc + 4];
	new_argv[0] = "gasnetrun_ibv";
	new_argv[1] = "-n";
	new_argv[2] = np_str;
	for(int i = 0; i < *argc; i++) new_argv[i+3] = (*argv)[i];
	new_argv[*argc + 3] = 0;
	execvp(new_argv[0], (char **)new_argv);
	// should never get here...
	perror("execvp");
      }
	  
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
      hcount += LockRequestMessage::add_handler_entries(&handlers[hcount]);
      hcount += LockReleaseMessage::add_handler_entries(&handlers[hcount]);
      hcount += LockGrantMessage::add_handler_entries(&handlers[hcount]);
      hcount += EventSubscribeMessage::add_handler_entries(&handlers[hcount]);
      hcount += EventTriggerMessage::add_handler_entries(&handlers[hcount]);
      hcount += RemoteMemAllocMessage::add_handler_entries(&handlers[hcount]);
      //hcount += TestMessage::add_handler_entries(&handlers[hcount]);
      //hcount += TestMessage2::add_handler_entries(&handlers[hcount]);

      unsigned shared_mem_size = 32;
      CHECK_GASNET( gasnet_attach(handlers, hcount, (shared_mem_size << 20), 0) );

      pthread_t poll_thread;
      CHECK_PTHREAD( pthread_create(&poll_thread, 0, gasnet_poll_thread_loop, 0) );
	
      //gasnet_seginfo_t seginfos = new gasnet_seginfo_t[num_nodes];
      //CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );

      Runtime *r = Runtime::runtime = new Runtime;
      r->nodes = new Node[gasnet_nodes()];
      
      r->global_memory = new GASNetMemory(ID(ID::ID_MEMORY, 0, ID::ID_GLOBAL_MEM, 0).convert<Memory>());

      Node *n = &r->nodes[gasnet_mynode()];

      NodeAnnounceData announce_data;

      unsigned num_local_procs = 1;

      announce_data.node_id = gasnet_mynode();
      announce_data.num_procs = num_local_procs;
      announce_data.num_memories = 1;

      // create local processors
      n->processors.resize(num_local_procs);

      for(unsigned i = 0; i < num_local_procs; i++) {
	n->processors[i] = new LocalProcessor(ID(ID::ID_PROCESSOR, gasnet_mynode(), i).convert<Processor>(), 
					      i);
	//local_procs[i]->start();
	//machine->add_processor(new LocalProcessor(local_procs[i]));
      }

      // create local memory
      Memory m = ID(ID::ID_MEMORY, gasnet_mynode(), 0).convert<Memory>();
      n->memories.resize(1);
      n->memories[0] = new LocalCPUMemory(m, 16 << 20);

      // now announce ourselves to everyone else
      for(int i = 0; i < gasnet_nodes(); i++)
	if(i != gasnet_mynode())
	  NodeAnnounceMessage::request(i, announce_data);

      // wait until we hear from everyone else?
      while(announcements_received < (gasnet_nodes() - 1))
	gasnet_AMPoll();

      printf("node %d has received all of its announcements\n", gasnet_mynode());

      for(int i = 0; i < gasnet_nodes(); i++) {
	unsigned np = Runtime::runtime->nodes[i].processors.size();
	unsigned nm = Runtime::runtime->nodes[i].memories.size();

	for(unsigned p = 0; p < np; p++)
	  procs.insert(ID(ID::ID_PROCESSOR, i, p).convert<Processor>());

	for(unsigned m = 0; m < nm; m++)
	  memories.insert(ID(ID::ID_MEMORY, i, m).convert<Memory>());

	for(unsigned p = 0; p < np; p++) {
	  Processor pid = ID(ID::ID_PROCESSOR, i, p).convert<Processor>();

	  for(unsigned m = 0; m < nm; m++) {
	    Memory mid = ID(ID::ID_MEMORY, i, m).convert<Memory>();
	    visible_memories_from_procs[pid].insert(mid);
	    visible_memories_from_procs[pid].insert(Runtime::runtime->global_memory->me);
	    visible_procs_from_memory[mid].insert(pid);
	    visible_procs_from_memory[Runtime::runtime->global_memory->me].insert(pid);
	    printf("P:%x <-> M:%x\n", pid.id, mid.id);
	  }
	}
      }	

      the_machine = this;
    }

    Machine::~Machine(void)
    {
      gasnet_exit(0);
    }

    Processor::Kind Machine::get_processor_kind(Processor p) const
    {
      return p.impl()->kind;
    }

    size_t Machine::get_memory_size(const Memory m) const
    {
      return m.impl()->size;
    }

    void Machine::run(Processor::TaskFuncID task_id /*= 0*/,
		      RunStyle style /*= ONE_TASK_ONLY*/,
		      const void *args /*= 0*/, size_t arglen /*= 0*/)
    {
      if(task_id != 0 && 
	 ((style != ONE_TASK_ONLY) || 
	  (gasnet_mynode() == 0))) {//(gasnet_nodes()-1)))) {
	const std::vector<Processor::Impl *>& local_procs = Runtime::runtime->nodes[gasnet_mynode()].processors;
	for(std::vector<Processor::Impl *>::const_iterator it = local_procs.begin();
	    it != local_procs.end();
	    it++) {
	  (*it)->spawn_task(task_id, args, arglen, 
			    Event::NO_EVENT, Event::NO_EVENT);
	  if(style != ONE_TASK_PER_PROC) break;
	}
      }

      // wait for idle-ness somehow?
      sleep(1000);
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
