#include "lowlevel.h"

#include <assert.h>

#define GASNET_PAR
#include <gasnet.h>

#define GASNETT_THREAD_SAFE
#include <gasnet_tools.h>

#include "activemsg.h"

#ifdef CHECK_REENTRANT_MESSAGES
GASNETT_THREADKEY_DEFINE(in_handler);
#endif
GASNETT_THREADKEY_DEFINE(cur_thread);

#include <pthread.h>
#include <string.h>

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

namespace RegionRuntime {
  namespace LowLevel {
    /*static*/ LogLevel Logger::log_level;
    /*static*/ std::vector<bool> Logger::log_cats_enabled;
    /*static*/ std::map<std::string, int> Logger::categories_by_name;
    /*static*/ std::vector<std::string> Logger::categories_by_id;

    /*static*/ void Logger::init(int argc, const char *argv[])
    {
      // default (for now) is to spew everything
      log_level = LEVEL_INFO;
      for(std::vector<bool>::iterator it = log_cats_enabled.begin();
	  it != log_cats_enabled.end();
	  it++)
	(*it) = true;

      for(int i = 1; i < argc; i++) {
	if(!strcmp(argv[i], "-level")) {
	  log_level = (LogLevel)atoi(argv[++i]);
	  continue;
	}

	if(!strcmp(argv[i], "-cat")) {
	  const char *p = argv[++i];

	  if(*p == '*') {
	    p++;
	  } else {
	    // turn off all the bits and then we'll turn on only what's requested
	    for(std::vector<bool>::iterator it = log_cats_enabled.begin();
		it != log_cats_enabled.end();
		it++)
	      (*it) = false;
	  }

	  while(*p == ',') p++;
	  while(*p) {
	    bool enable = true;
	    if(*p == '-') {
	      enable = false;
	      p++;
	    }
	    const char *p2 = p; while(*p2 && (*p2 != ',')) p2++;
	    std::string name(p, p2);
	    std::map<std::string, int>::iterator it = categories_by_name.find(name);
	    if(it == categories_by_name.end()) {
	      fprintf(stderr, "unknown log category '%s'!\n", name.c_str());
	      exit(1);
	    }

	    log_cats_enabled[it->second] = enable;

	    p = p2;
	    while(*p == ',') p++;
	  }
	}
	continue;
      }
#if 1
      printf("logger settings: level=%d cats=", log_level);
      bool first = true;
      for(unsigned i = 0; i < log_cats_enabled.size(); i++)
	if(log_cats_enabled[i]) {
	  if(!first) printf(",");
	  first = false;
	  printf("%s", categories_by_id[i].c_str());
	}
      printf("\n");
#endif
    }

    /*static*/ void Logger::logvprintf(LogLevel level, int category, const char *fmt, va_list args)
    {
      char buffer[200];
      sprintf(buffer, "[%d - %lx] {%d}{%s}: ",
	      gasnet_mynode(), pthread_self(), level, categories_by_id[category].c_str());
      int len = strlen(buffer);
      vsnprintf(buffer+len, 199-len, fmt, args);
      strcat(buffer, "\n");
      fputs(buffer, stderr);
    }

    static Logger::Category log_mutex("mutex");

    class AutoHSLLock {
    public:
      AutoHSLLock(gasnet_hsl_t &mutex) : mutexp(&mutex) 
      { 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK IN %p", mutexp);
	//printf("[%d] MUTEX LOCK IN %p\n", gasnet_mynode(), mutexp);
	gasnet_hsl_lock(mutexp); 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK HELD %p", mutexp);
	//printf("[%d] MUTEX LOCK HELD %p\n", gasnet_mynode(), mutexp);
      }
      AutoHSLLock(gasnet_hsl_t *_mutexp) : mutexp(_mutexp) 
      { 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK IN %p", mutexp);
	//printf("[%d] MUTEX LOCK IN %p\n", gasnet_mynode(), mutexp);
	gasnet_hsl_lock(mutexp); 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK HELD %p", mutexp);
	//printf("[%d] MUTEX LOCK HELD %p\n", gasnet_mynode(), mutexp);
      }
      ~AutoHSLLock(void) 
      { 
	gasnet_hsl_unlock(mutexp);
	log_mutex(LEVEL_SPEW, "MUTEX LOCK OUT %p", mutexp);
	//printf("[%d] MUTEX LOCK OUT %p\n", gasnet_mynode(), mutexp);
      }
    protected:
      gasnet_hsl_t *mutexp;
    };



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
      CREATE_ALLOC_MSGID,
      CREATE_ALLOC_RPLID,
      CREATE_INST_MSGID,
      CREATE_INST_RPLID,
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

      bool is_locked(unsigned check_mode, bool excl_ok);
    };

    template <class T>
    class StaticAccess {
    public:
      typedef typename T::StaticData StaticData;

      // if already_valid, just check that data is already valid
      StaticAccess(T* thing_with_data, bool already_valid = false)
	: data(&thing_with_data->locked_data)
      {
	if(already_valid) {
	  assert(data->valid);
	} else {
	  if(!data->valid) {
	    // get a valid copy of the static data by taking and then releasing
	    //  a shared lock
	    thing_with_data->lock.lock(1, false).wait();
	    thing_with_data->lock.unlock();
	    assert(data->valid);
	  }
	}
      }

      ~StaticAccess(void) {}

      const StaticData *operator->(void) { return data; }

    protected:
      StaticData *data;
    };

    template <class T>
    class SharedAccess {
    public:
      typedef typename T::CoherentData CoherentData;

      // if already_held, just check that it's held (if in debug mode)
      SharedAccess(T* thing_with_data, bool already_held = false)
	: data(&thing_with_data->locked_data), lock(&thing_with_data->lock)
      {
	if(already_held) {
	  assert(lock->is_locked(1, true));
	} else {
	  lock->lock(1, false).wait();
	}
      }

      ~SharedAccess(void)
      {
	lock->unlock();
      }

      const CoherentData *operator->(void) { return data; }

    protected:
      CoherentData *data;
      Lock::Impl *lock;
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
	locked_data.valid = true;
	locked_data.num_elmts = _num_elmts;
	locked_data.elmt_size = _elmt_size;
	lock.init(ID(me).convert<Lock>(), ID(me).node());
	lock.set_local_data(&locked_data);
      }

      // this version is called when we create a proxy for a remote region
      Impl(RegionMetaDataUntyped _me)
	: me(_me)
      {
	locked_data.valid = false;
	locked_data.num_elmts = 0;
	locked_data.elmt_size = 0;  // automatically ask for shared lock to fill these in?
	lock.init(ID(me).convert<Lock>(), ID(me).node());
	lock.set_local_data(&locked_data);
      }

      ~Impl(void) {}

      size_t instance_size(void)
      {
	StaticAccess<RegionMetaDataUntyped::Impl> data(this);
	assert(data->num_elmts > 0);
	assert(data->elmt_size > 0);
	size_t bytes = data->num_elmts * data->elmt_size;
	return bytes;
      }

      RegionMetaDataUntyped me;
      Lock::Impl lock;

      struct StaticData {
	bool valid;
	size_t num_elmts, elmt_size;
      };

      StaticData locked_data;
    };

    class RegionAllocatorUntyped::Impl {
    public:
      Impl(RegionAllocatorUntyped _me, RegionMetaDataUntyped _region, Memory _memory, int _mask_start);

      ~Impl(void);

      unsigned alloc_elements(unsigned count = 1);

      void free_elements(unsigned ptr, unsigned count = 1);

      struct StaticData {
	RegionMetaDataUntyped region;
	Memory memory;
	int mask_start;
      };

      RegionAllocatorUntyped me;
      StaticData data;

    protected:
      ElementMask avail_elmts, changed_elmts;
    };

    class RegionInstanceUntyped::Impl {
    public:
      Impl(RegionInstanceUntyped _me, RegionMetaDataUntyped _region, Memory _memory, int _offset)
	: me(_me), memory(_memory)
      {
	locked_data.valid = true;
	locked_data.region = _region;
	locked_data.offset = _offset;
	lock.init(ID(me).convert<Lock>(), ID(me).node());
	lock.set_local_data(&locked_data);
      }

      // when we auto-create a remote instance, we don't know region/offset
      Impl(RegionInstanceUntyped _me, Memory _memory)
	: me(_me), memory(_memory)
      {
	locked_data.valid = false;
	locked_data.region = RegionMetaDataUntyped::NO_REGION;
	locked_data.offset = -1;
	lock.init(ID(me).convert<Lock>(), ID(me).node());
	lock.set_local_data(&locked_data);
      }

      ~Impl(void) {}

      void get_bytes(unsigned ptr_value, void *dst, size_t size);
      void put_bytes(unsigned ptr_value, const void *src, size_t size);

      static void copy(RegionInstanceUntyped src, 
		       RegionInstanceUntyped target,
		       size_t bytes_to_copy,
		       Event after_copy = Event::NO_EVENT);

    public: //protected:
      friend class RegionInstanceUntyped;

      RegionInstanceUntyped me;
      Memory memory;

      struct StaticData {
	bool valid;
	RegionMetaDataUntyped region;
	int offset;
      } locked_data;

      Lock::Impl lock;
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
      //void wait(Event::gen_t needed_gen);

      // creates an event that won't trigger until all input events have
      static Event merge_events(const std::set<Event>& wait_for);
      static Event merge_events(Event ev1, Event ev2,
				Event ev3 = NO_EVENT, Event ev4 = NO_EVENT,
				Event ev5 = NO_EVENT, Event ev6 = NO_EVENT);

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

    static Logger::Category log_event("event");

    void handle_event_subscribe(EventSubscribeArgs args)
    {
      log_event(LEVEL_DEBUG, "event subscription: node=%d event=%x/%d",
		args.node, args.event.id, args.event.gen);

      Event::Impl *impl = args.event.impl();

      // early-out case: if we can see the generation needed has already
      //  triggered, signal without taking the mutex
      unsigned stale_gen = impl->generation;
      if(stale_gen >= args.event.gen) {
	log_event(LEVEL_DEBUG, "event subscription early-out: node=%d event=%x/%d (<= %d)",
		  args.node, args.event.id, args.event.gen, stale_gen);
	Event e = args.event;
	e.gen = stale_gen;
	EventTriggerMessage::request(args.node, e);
	return;
      }

      {
	AutoHSLLock a(impl->mutex);

	// now that we have the lock, check the needed generation again
	if(impl->generation >= args.event.gen) {
	  log_event(LEVEL_DEBUG, "event subscription already done: node=%d event=%x/%d (<= %d)",
		    args.node, args.event.id, args.event.gen, impl->generation);
	  Event e = args.event;
	  e.gen = impl->generation;
	  EventTriggerMessage::request(args.node, e);
	} else {
	  // nope - needed generation hasn't happened yet, so add this node to
	  //  the mask
	  log_event(LEVEL_DEBUG, "event subscription recorded: node=%d event=%x/%d (> %d)",
		    args.node, args.event.id, args.event.gen, impl->generation);
	  impl->remote_waiters |= (1ULL << args.node);
	}
      }
    }

    void handle_event_trigger(Event event)
    {
      log_event(LEVEL_DEBUG, "Remote trigger of event %x/%d!", event.id, event.gen);
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

    class EventMerger : public Event::Impl::EventWaiter {
    public:
      EventMerger(Event _finish_event)
	: count_needed(1), finish_event(_finish_event)
      {
	gasnet_hsl_init(&mutex);
      }

      void add_event(Event wait_for)
      {
	if(wait_for.has_triggered()) return; // early out
	{
	  // step 1: increment our count first - we can't hold the lock while
	  //   we add a listener to the 'wait_for' event (since it might trigger
	  //   instantly and call our count-decrementing function), and we
	  //   need to make sure all increments happen before corresponding
	  //   decrements
	  AutoHSLLock a(mutex);
	  count_needed++;
	}
	// step 2: enqueue ourselves on the input event
	wait_for.impl()->add_waiter(wait_for, this);
      }

      // arms the merged event once you're done adding input events - just
      //  decrements the count for the implicit 'init done' event
      void arm(void)
      {
	event_triggered();
      }

      virtual void event_triggered(void)
      {
	bool last_trigger = false;
	{
	  AutoHSLLock a(mutex);
	  count_needed--;
	  if(count_needed == 0) last_trigger = true;
	}
	// actually do triggering outside of lock (maybe not necessary, but
	//  feels safer :)
	if(last_trigger)
	  finish_event.impl()->trigger(finish_event.gen, true);
      }

    protected:
      unsigned count_needed;
      Event finish_event;
      gasnet_hsl_t mutex;
    };

    // creates an event that won't trigger until all input events have
    /*static*/ Event Event::Impl::merge_events(const std::set<Event>& wait_for)
    {
      // scan through events to see how many exist/haven't fired - we're
      //  interested in counts of 0, 1, or 2+ - also remember the first
      //  event we saw for the count==1 case
      int wait_count = 0;
      Event first_wait;
      for(std::set<Event>::const_iterator it = wait_for.begin();
	  (it != wait_for.end()) && (wait_count < 2);
	  it++)
	if(!(*it).has_triggered()) {
	  if(!wait_count) first_wait = *it;
	  wait_count++;
	}

      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if(wait_count == 1) return first_wait;

      // counts of 2+ require building a new event and a merger to trigger it
      Event finish_event = Event::Impl::create_event();
      EventMerger *m = new EventMerger(finish_event);

      for(std::set<Event>::const_iterator it = wait_for.begin();
	  (it != wait_for.end()) && (wait_count < 2);
	  it++)
	m->add_event(*it);

      // once they're all added - arm the thing (it might go off immediately)
      m->arm();

      return finish_event;
    }

    /*static*/ Event Event::Impl::merge_events(Event ev1, Event ev2,
					       Event ev3 /*= NO_EVENT*/, Event ev4 /*= NO_EVENT*/,
					       Event ev5 /*= NO_EVENT*/, Event ev6 /*= NO_EVENT*/)
    {
      // scan through events to see how many exist/haven't fired - we're
      //  interested in counts of 0, 1, or 2+ - also remember the first
      //  event we saw for the count==1 case
      int wait_count = 0;
      Event first_wait;
      if(!ev6.has_triggered()) { first_wait = ev6; wait_count++; }
      if(!ev5.has_triggered()) { first_wait = ev5; wait_count++; }
      if(!ev4.has_triggered()) { first_wait = ev4; wait_count++; }
      if(!ev3.has_triggered()) { first_wait = ev3; wait_count++; }
      if(!ev2.has_triggered()) { first_wait = ev2; wait_count++; }
      if(!ev1.has_triggered()) { first_wait = ev1; wait_count++; }

      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if(wait_count == 1) return first_wait;

      // counts of 2+ require building a new event and a merger to trigger it
      Event finish_event = Event::Impl::create_event();
      EventMerger *m = new EventMerger(finish_event);

      m->add_event(ev1);
      m->add_event(ev2);
      m->add_event(ev3);
      m->add_event(ev4);
      m->add_event(ev5);
      m->add_event(ev6);

      // once they're all added - arm the thing (it might go off immediately)
      m->arm();

      return finish_event;
    }

    // creates an event that won't trigger until all input events have
    /*static*/ Event Event::merge_events(const std::set<Event>& wait_for)
    {
      return Event::Impl::merge_events(wait_for);
    }

    /*static*/ Event Event::merge_events(Event ev1, Event ev2,
					 Event ev3 /*= NO_EVENT*/, Event ev4 /*= NO_EVENT*/,
					 Event ev5 /*= NO_EVENT*/, Event ev6 /*= NO_EVENT*/)
    {
      return Event::Impl::merge_events(ev1, ev2, ev3, ev4, ev5, ev6);
    }

    /*static*/ UserEvent UserEvent::create_user_event(void)
    {
      Event e = Event::Impl::create_event();
      assert(e.id != 0);
      UserEvent u;
      u.id = e.id;
      u.gen = e.gen;
      return u;
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
	  log_event(LEVEL_SPEW, "event reused: event=%x/%d", ev.id, ev.gen);
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
      log_event(LEVEL_SPEW, "event created: event=%x/%d", ev.id, ev.gen);
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
	  log_event(LEVEL_DEBUG, "event not ready: event=%x/%d owner=%d gen=%d subscr=%d",
		    event.id, event.gen, owner, generation, gen_subscribed);
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
      return (needed_gen <= generation);
    }

#if 0
    void Event::Impl::wait(Event::gen_t needed_gen)
    {
      if(has_triggered(needed_gen)) return; // early out

      // figure out which thread we are - better be a local CPU thread!
      void *ptr = gasnet_threadkey_get(cur_thread);
      assert(ptr != 0);
      LocalProcessor::Thread *thr = (LocalProcessor::Thread *)ptr;
      thr->sleep_on_event(this, needed_gen);
    }
#endif
    
    void Event::Impl::trigger(Event::gen_t gen_triggered, bool local_trigger)
    {
      log_event(LEVEL_SPEW, "event triggered: event=%x/%d", me.id, gen_triggered);
      //printf("[%d] TRIGGER %x/%d\n", gasnet_mynode(), me.id, gen_triggered);
      std::deque<EventWaiter *> to_wake;
      {
	//printf("[%d] TRIGGER MUTEX IN %x/%d\n", gasnet_mynode(), me.id, gen_triggered);
	AutoHSLLock a(mutex);
	//printf("[%d] TRIGGER MUTEX HOLD %x/%d\n", gasnet_mynode(), me.id, gen_triggered);

	//printf("[%d] TRIGGER GEN: %x/%d->%d\n", gasnet_mynode(), me.id, generation, gen_triggered);
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

    Logger::Category log_lock("lock");

    /*static*/ void /*Lock::Impl::*/handle_lock_request(LockRequestArgs args)
    {
      Lock::Impl *impl = args.lock.impl();

      log_lock(LEVEL_DEBUG, "lock request: lock=%x, node=%d, mode=%d",
	       args.lock.id, args.node, args.mode);

      // can't send messages while holding mutex, so remember args and who
      //  (if anyone) to send to
      int req_forward_target = -1;
      int grant_target = -1;
      LockGrantArgs g_args;

      do {
	AutoHSLLock a(impl->mutex);

	// case 1: we don't even own the lock any more - pass the request on
	//  to whoever we think the owner is
	if(impl->owner != gasnet_mynode()) {
	  // can reuse the args we were given
	  req_forward_target = impl->owner;
	  break;
	}

	// case 2: we're the owner, and nobody is holding the lock, so grant
	//  it to the (original) requestor
	if((impl->count == 0) && (impl->remote_sharer_mask == 0)) {
	  assert(impl->remote_waiter_mask == 0);

	  log_lock(LEVEL_DEBUG, "granting lock request: lock=%x, node=%d, mode=%d",
		   args.lock.id, args.node, args.mode);
	  g_args.lock = args.lock;
	  g_args.mode = 0; // always give it exclusively for now
	  g_args.remote_waiter_mask = impl->remote_waiter_mask;
	  grant_target = args.node;

	  impl->owner = args.node;
	  break;
	}

	// case 3: we're the owner, but we can't grant the lock right now -
	//  just set a bit saying that the node is waiting and get back to
	//  work
	log_lock(LEVEL_DEBUG, "deferring lock request: lock=%x, node=%d, mode=%d (count=%d cmode=%d)",
		 args.lock.id, args.node, args.mode, impl->count, impl->mode);
	impl->remote_waiter_mask |= (1ULL << args.node);
      } while(0);

      if(req_forward_target != -1)
	LockRequestMessage::request(req_forward_target, args);

      if(grant_target != -1)
	LockGrantMessage::request(grant_target, g_args,
				  impl->local_data, impl->local_data_size);
    }

    /*static*/ void /*Lock::Impl::*/handle_lock_release(LockReleaseArgs args)
    {
      assert(0);
    }

    void handle_lock_grant(LockGrantArgs args, const void *data, size_t datalen)
    {
      log_lock(LEVEL_DEBUG, "lock request granted: lock=%x mode=%d mask=%lx",
	       args.lock.id, args.mode, args.remote_waiter_mask);

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
	log_lock(LEVEL_DEBUG, "unlock trigger: lock=%x event=%x/%d",
		 args.lock.id, (*it).id, (*it).gen);
	(*it).impl()->trigger((*it).gen, true);
      }
    }

    Event Lock::Impl::lock(unsigned new_mode, bool exclusive,
			   Event after_lock /*= Event::NO_EVENT*/)
    {
      log_lock(LEVEL_DEBUG, "local lock request: lock=%x mode=%d excl=%d event=%x/%d",
	       me.id, new_mode, exclusive, after_lock.id, after_lock.gen);

      // deferred lock case
      if(after_lock.exists()) {
	assert(0);
      }

      // collapse exclusivity into mode
      if(exclusive) new_mode = MODE_EXCL;

      bool got_lock = false;
      int lock_request_target = -1;
      LockRequestArgs args;

      {
	AutoHSLLock a(mutex); // hold mutex on lock while we check things

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
	    log_lock(LEVEL_DEBUG, "requesting lock: lock=%x node=%d mode=%d",
		     me.id, owner, new_mode);
	    args.node = gasnet_mynode();
	    args.lock = me;
	    args.mode = new_mode;
	    lock_request_target = owner;
	    // don't actually send message here because we're holding the
	    //  lock's mutex, which'll be bad if we get a message related to
	    //  this lock inside gasnet calls
	  
	    requested = true;
	  }
	}

	// if we didn't get the lock, put our event on the queue of local
	//  waiters - create an event if we weren't given one to use
	if(!got_lock) {
	  if(!after_lock.exists())
	    after_lock = Event::Impl::create_event();
	  local_waiters[new_mode].push_back(after_lock);
	}
      }

      if(lock_request_target != -1)
	LockRequestMessage::request(lock_request_target, args);

      // if we got the lock, trigger an event if we were given one
      if(got_lock && after_lock.exists()) 
	after_lock.impl()->trigger(after_lock.gen, true);

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
      log_lock(LEVEL_DEBUG, "lock going to local waiter: size=%zd first=%d(%zd)",
	       local_waiters.size(), 
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

      int release_target = -1;
      LockReleaseArgs r_args;

      int grant_target = -1;
      LockGrantArgs g_args;

      do {
	log_lock(LEVEL_DEBUG, "unlock: lock=%x count=%d mode=%d share=%lx wait=%lx",
		 me.id, count, mode, remote_sharer_mask, remote_waiter_mask);
	AutoHSLLock a(mutex); // hold mutex on lock for entire function

	assert(count > 0);

	// if this isn't the last holder of the lock, just decrement count
	//  and return
	count--;
	if(count > 0) break;

	// case 1: if we were sharing somebody else's lock, tell them we're
	//  done
	if(owner != gasnet_mynode()) {
	  assert(mode != MODE_EXCL);
	  mode = 0;

	  r_args.node = gasnet_mynode();
	  r_args.lock = me;
	  release_target = owner;
	  break;
	}

	// case 2: we own the lock, so we can give it to another waiter
	//  (local or remote)
	bool any_local = select_local_waiters(to_wake);

	if(!any_local && (remote_waiter_mask != 0)) {
	  // nobody local wants it, but another node does
	  int new_owner = 0;
	  while(((remote_waiter_mask >> new_owner) & 1) == 0) new_owner++;

	  log_lock(LEVEL_DEBUG, "lock going to remote waiter: new=%d mask=%lx",
		   new_owner, remote_waiter_mask);

	  g_args.lock = me;
	  g_args.mode = 0; // TODO: figure out shared cases
	  g_args.remote_waiter_mask = remote_waiter_mask & ~(1ULL << new_owner);
	  grant_target = new_owner;

	  owner = new_owner;
	  remote_waiter_mask = 0;
	}
      } while(0);

      if(release_target != -1)
	LockReleaseMessage::request(release_target, r_args);

      if(grant_target != -1)
	LockGrantMessage::request(grant_target, g_args,
				  local_data, local_data_size);

      for(std::deque<Event>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	log_lock(LEVEL_DEBUG, "unlock trigger: lock=%x event=%x/%d",
		 me.id, (*it).id, (*it).gen);
	(*it).impl()->trigger((*it).gen, true);
      }
    }

    bool Lock::Impl::is_locked(unsigned check_mode, bool excl_ok)
    {
      // checking the owner can be done atomically, so doesn't need mutex
      if(owner != gasnet_mynode()) return false;

      // conservative check on lock count also doesn't need mutex
      if(count == 0) return false;

      // a careful check of the lock mode and count does require the mutex
      bool held;
      {
	AutoHSLLock a(mutex);

	held = ((count > 0) &&
		((mode == check_mode) || ((mode == 0) && excl_ok)));
      }

      return held;
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
      //printf("LOCK(%x, %d, %d, %x) -> ", id, mode, exclusive, wait_on.id);
      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(wait_on.has_triggered()) {
	Event e = impl()->lock(mode, exclusive);
	//printf("(%x/%d)\n", e.id, e.gen);
	return e;
      } else {
	Event after_lock = Event::Impl::create_event();
	wait_on.impl()->add_waiter(wait_on, new DeferredLockRequest(*this, mode, exclusive, after_lock));
	//printf("*(%x/%d)\n", after_lock.id, after_lock.gen);
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
      enum MemoryKind {
	MKIND_SYSMEM,  // directly accessible from CPU
	MKIND_GASNET,  // accessible via GASnet RDMA
	MKIND_REMOTE,  // not accessible
	MKIND_GPUFB,   // GPU framebuffer memory (accessible via cudaMemcpy)
	MKIND_ZEROCOPY, // CPU memory, pinned for GPU access
      };

      Impl(Memory _me, size_t _size, MemoryKind _kind)
	: me(_me), size(_size), kind(_kind)
      {
	gasnet_hsl_init(&mutex);
      }

      unsigned add_allocator(RegionAllocatorUntyped::Impl *a);
      unsigned add_instance(RegionInstanceUntyped::Impl *i);

      RegionAllocatorUntyped::Impl *get_allocator(RegionAllocatorUntyped a);
      RegionInstanceUntyped::Impl *get_instance(RegionInstanceUntyped i);

      RegionAllocatorUntyped create_allocator_local(RegionMetaDataUntyped r,
						    size_t bytes_needed);
      RegionInstanceUntyped create_instance_local(RegionMetaDataUntyped r,
						  size_t bytes_needed);

      RegionAllocatorUntyped create_allocator_remote(RegionMetaDataUntyped r,
						     size_t bytes_needed);
      RegionInstanceUntyped create_instance_remote(RegionMetaDataUntyped r,
						   size_t bytes_needed);

      virtual RegionAllocatorUntyped create_allocator(RegionMetaDataUntyped r,
						      size_t bytes_needed) = 0;
      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed) = 0;

      void destroy_allocator(RegionAllocatorUntyped a, bool local_destroy);
      void destroy_instance(RegionInstanceUntyped i, bool local_destroy);

      virtual int alloc_bytes(size_t size) = 0;
      virtual void free_bytes(int offset, size_t size) = 0;

      virtual void get_bytes(unsigned offset, void *dst, size_t size) = 0;
      virtual void put_bytes(unsigned offset, const void *src, size_t size) = 0;

      virtual void *get_direct_ptr(unsigned offset, size_t size) = 0;

    public:
      Memory me;
      size_t size;
      MemoryKind kind;
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
      //printf("[%d] handling remote alloc of size %zd\n", gasnet_mynode(), args.size);
      int result = args.memory.impl()->alloc_bytes(args.size);
      //printf("[%d] remote alloc will return %d\n", gasnet_mynode(), result);
      return result;
    }

    typedef ActiveMessageShortReply<REMOTE_MALLOC_MSGID, REMOTE_MALLOC_RPLID,
				    RemoteMemAllocArgs, int,
				    handle_remote_mem_alloc> RemoteMemAllocMessage;

    class LocalCPUMemory : public Memory::Impl {
    public:
      LocalCPUMemory(Memory _me, size_t _size) 
	: Memory::Impl(_me, _size, MKIND_SYSMEM)
      {
	base = new char[_size];
	free_blocks[0] = _size;
      }

      virtual ~LocalCPUMemory(void)
      {
	delete[] base;
      }

      virtual RegionAllocatorUntyped create_allocator(RegionMetaDataUntyped r,
						      size_t bytes_needed)
      {
	return create_allocator_local(r, bytes_needed);
      }

      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed)
      {
	return create_instance_local(r, bytes_needed);
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
	: Memory::Impl(_me, _size, MKIND_REMOTE)
      {
      }

      virtual RegionAllocatorUntyped create_allocator(RegionMetaDataUntyped r,
						      size_t bytes_needed)
      {
	return create_allocator_remote(r, bytes_needed);
      }

      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed)
      {
	return create_instance_remote(r, bytes_needed);
      }

      virtual int alloc_bytes(size_t size)
      {
	// RPC over to owner's node for allocation

	RemoteMemAllocArgs args;
	args.memory = me;
	args.size = size;
	int retval = RemoteMemAllocMessage::request(ID(me).node(), args);
	//printf("got: %d\n", retval);
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
	: Memory::Impl(_me, 0 /* we'll calculate it below */, MKIND_GASNET)
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

      virtual RegionAllocatorUntyped create_allocator(RegionMetaDataUntyped r,
						      size_t bytes_needed)
      {
	if(gasnet_mynode() == 0) {
	  return create_allocator_local(r, bytes_needed);
	} else {
	  return create_allocator_remote(r, bytes_needed);
	}
      }

      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed)
      {
	if(gasnet_mynode() == 0) {
	  return create_instance_local(r, bytes_needed);
	} else {
	  return create_instance_remote(r, bytes_needed);
	}
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
	  //printf("got: %d\n", retval);
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

    RegionAllocatorUntyped Memory::Impl::create_allocator_local(RegionMetaDataUntyped r,
								size_t bytes_needed)
    {
      int mask_start = alloc_bytes(bytes_needed);

      // now find/make an available index to store this in
      unsigned index;
      {
	AutoHSLLock a(mutex);

	unsigned size = allocators.size();
	for(index = 0; index < size; index++)
	  if(!allocators[index]) {
	    allocators[index] = (RegionAllocatorUntyped::Impl *)1;
	    break;
	  }

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

    RegionInstanceUntyped Memory::Impl::create_instance_local(RegionMetaDataUntyped r,
							      size_t bytes_needed)
    {
      int inst_offset = alloc_bytes(bytes_needed);

      // find/make an available index to store this in
      unsigned index;
      {
	AutoHSLLock a(mutex);

	unsigned size = instances.size();
	for(index = 0; index < size; index++)
	  if(!instances[index]) {
	    instances[index] = (RegionInstanceUntyped::Impl *)1;
	    break;
	  }

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

    struct CreateAllocatorArgs {
      Memory m;
      RegionMetaDataUntyped r;
      size_t bytes_needed;
    };

    struct CreateAllocatorResp {
      RegionAllocatorUntyped a;
      int mask_start;
    };

    CreateAllocatorResp handle_create_allocator(CreateAllocatorArgs args)
    {
      CreateAllocatorResp resp;
      resp.a = args.m.impl()->create_allocator(args.r, args.bytes_needed);
      resp.mask_start = resp.a.impl()->data.mask_start;
      return resp;
    }

    typedef ActiveMessageShortReply<CREATE_ALLOC_MSGID, CREATE_ALLOC_RPLID,
				    CreateAllocatorArgs, CreateAllocatorResp,
				    handle_create_allocator> CreateAllocatorMessage;

    RegionAllocatorUntyped Memory::Impl::create_allocator_remote(RegionMetaDataUntyped r,
								 size_t bytes_needed)
    {
      CreateAllocatorArgs args;
      args.m = me;
      args.r = r;
      args.bytes_needed = bytes_needed;
      CreateAllocatorResp resp = CreateAllocatorMessage::request(ID(me).node(), args);
      RegionAllocatorUntyped::Impl *a_impl = new RegionAllocatorUntyped::Impl(resp.a, r, me, resp.mask_start);
      unsigned index = ID(resp.a).index_l();
      // resize array if needed
      if(index >= allocators.size()) {
	AutoHSLLock a(mutex);
	if(index >= allocators.size())
	  for(unsigned i = allocators.size(); i <= index; i++)
	    allocators.push_back(0);
      }
      allocators[index] = a_impl;
      return resp.a;
    }

    struct CreateInstanceArgs {
      Memory m;
      RegionMetaDataUntyped r;
      size_t bytes_needed;
    };

    struct CreateInstanceResp {
      RegionInstanceUntyped i;
      int inst_offset;
    };

    CreateInstanceResp handle_create_instance(CreateInstanceArgs args)
    {
      CreateInstanceResp resp;
      resp.i = args.m.impl()->create_instance(args.r, args.bytes_needed);
      resp.inst_offset = resp.i.impl()->locked_data.offset; // TODO: Static
      return resp;
    }

    typedef ActiveMessageShortReply<CREATE_INST_MSGID, CREATE_INST_RPLID,
				    CreateInstanceArgs, CreateInstanceResp,
				    handle_create_instance> CreateInstanceMessage;

    Logger::Category log_inst("inst");

    RegionInstanceUntyped Memory::Impl::create_instance_remote(RegionMetaDataUntyped r,
							       size_t bytes_needed)
    {
      CreateInstanceArgs args;
      args.m = me;
      args.r = r;
      args.bytes_needed = bytes_needed;
      log_inst(LEVEL_DEBUG, "creating remote instance: node=%d", ID(me).node());
      CreateInstanceResp resp = CreateInstanceMessage::request(ID(me).node(), args);
      log_inst(LEVEL_DEBUG, "created remote instance: inst=%x offset=%d", resp.i.id, resp.inst_offset);
      RegionInstanceUntyped::Impl *i_impl = new RegionInstanceUntyped::Impl(resp.i, r, me, resp.inst_offset);
      unsigned index = ID(resp.i).index_l();
      // resize array if needed
      if(index >= instances.size()) {
	AutoHSLLock a(mutex);
	if(index >= instances.size()) {
	  log_inst(LEVEL_DEBUG, "resizing instance array: mem=%x old=%zd new=%d",
		   me.id, instances.size(), index+1);
	  for(unsigned i = instances.size(); i <= index; i++)
	    instances.push_back(0);
	}
      }
      instances[index] = i_impl;
      return resp.i;
    }

#if 0
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
#endif

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

    void Memory::Impl::destroy_allocator(RegionAllocatorUntyped i, bool local_destroy)
    {
      ID id(i);

      // TODO: actually free corresponding storage

      unsigned index = id.index_l();
      assert(index < allocators.size());
      delete allocators[index];
      allocators[index] = 0;
    }

    void Memory::Impl::destroy_instance(RegionInstanceUntyped i, bool local_destroy)
    {
      return; // TODO: FIX!
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
			      //std::set<RegionInstanceUntyped> instances_needed,
			      Event start_event, Event finish_event) = 0;

    public:
      Processor me;
      Processor::Kind kind;
    };

    Logger::Category log_task("task");

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
	  log_task(LEVEL_DEBUG, "task start: %d (%p) (%s)", func_id, fptr, argstr);
	  (*fptr)(args, arglen, proc->me);
	  log_task(LEVEL_DEBUG, "task end: %d (%p) (%s)", func_id, fptr, argstr);
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
      class Thread : public Event::Impl::EventWaiter {
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
	  assert(new_task);
	  task = new_task;
	  gasnett_cond_signal(&condvar);
	}

	void resume_task(void)
	{
	  // assumes proc's mutex already held
	  state = STATE_RUN;
	  gasnett_cond_signal(&condvar);
	}

	virtual void event_triggered(void)
	{
	  AutoHSLLock a(proc->mutex);

	  // check for the instant trigger guard - if it's still set, the
	  //  thread didn't go to sleep yet, so just clear the guard rather
	  //  than moving the thread to the resumable list
	  if(instant_trigger_guard) {
	    instant_trigger_guard = false;
	  } else {
	    proc->resumable_threads.push_back(this);
	    proc->start_some_threads();
	  }
	}

	void sleep_on_event(Event wait_for)
	{
#define MULTIPLE_TASKS_PER_THREAD	  
#ifdef MULTIPLE_TASKS_PER_THREAD	  
	  // if we're going to wait, see if there's something useful
	  //  we can do in the meantime
	  while(1) {
	    if(wait_for.has_triggered()) return; // early out

	    AutoHSLLock a(proc->mutex);

	    proc->active_thread_count--;
	    log_task(LEVEL_DEBUG, "thread needs to wait on event: event=%x/%d",
		     wait_for.id, wait_for.gen);
	    Task *new_task = proc->select_task();
	    if(!new_task) {
	      proc->active_thread_count++; //put count back (we'll dec below)
	      break;  // nope, have to sleep
	    }

	    Task *old_task = task;
	    log_task(LEVEL_DEBUG, "thread task swap: old=%p new=%p",
		     old_task, new_task);
	    task = new_task;

	    // run task (without lock)
	    gasnet_hsl_unlock(&proc->mutex);
	    task->run();
	    gasnet_hsl_lock(&proc->mutex);

	    // TODO: delete task?
	    if(task == proc->idle_task) {
	      log_task(LEVEL_SPEW, "thread returned from idle task: proc=%x", proc->me.id);
	      proc->in_idle_task = false;
	    }

	    log_task(LEVEL_DEBUG, "thread returning to old task: old=%p new=%p",
		     old_task, new_task);
	    task = old_task;
	  }
#endif

	  // icky race conditions here - once we add ourselves as a waiter, 
	  //  the trigger could come right away (and need the lock), so we
	  //  have to set a flag (also save the event ID we're waiting on
	  //  for debug goodness), then add a waiter to the event and THEN
	  //  take our own lock and see if we still need to sleep
	  instant_trigger_guard = true;
	  suspend_event = wait_for;

	  wait_for.impl()->add_waiter(wait_for, this);

	  {
	    AutoHSLLock a(proc->mutex);

	    if(instant_trigger_guard) {
	      // guard is still active, so we can safely sleep
	      instant_trigger_guard = false;

	      // NOTE: while tempting, it's not OK to check the event's
	      //  triggeredness again here - it can result in an event_triggered()
	      //  sent to us without us going to sleep, which would be bad
	      log_task(LEVEL_DEBUG, "thread sleeping on event: thread=%p event=%x/%d",
		       this, wait_for.id, wait_for.gen);
	      // decrement the active thread count (check to see if somebody
	      //  else can run)
	      proc->active_thread_count--;
	      log_task(LEVEL_SPEW, "ATC = %d", proc->active_thread_count);
	      proc->start_some_threads();

	      if((proc->active_thread_count == 0) &&
		 (proc->avail_threads.size() == 0) &&
		 (proc->ready_tasks.size() > 0)) 
		log_task(LEVEL_INFO, "warning: all threads for proc=%x sleeping with tasks ready", proc->me.id);

	      // now sleep on our condition variable - we'll wake up after
	      //  we've been moved to the resumable list and chosen from there
	      state = STATE_SUSPEND;
	      fflush(stdout);
	      do {
		// guard against spurious wakeups
		gasnett_cond_wait(&condvar, &proc->mutex.lock);
	      } while(state == STATE_SUSPEND);
	      log_task(LEVEL_DEBUG, "thread done sleeping on event: thread=%p event=%x/%d",
		       this, wait_for.id, wait_for.gen);
	    } else {
	      log_task(LEVEL_DEBUG, "thread got instant trigger on event: thread=%p event=%x/%d",
		       this, wait_for.id, wait_for.gen);
	    }
	  }
	}

	static void *thread_main(void *args)
	{
	  Thread *me = (Thread *)args;

	  // first thing - take the lock and set our status
	  me->state = STATE_IDLE;

	  // stuff our pointer into TLS so Event::wait can find it
	  gasnett_threadkey_set(cur_thread, me);

	  LocalProcessor *proc = me->proc;
	  log_task(LEVEL_DEBUG, "worker thread ready: proc=%x", proc->me.id);
	  // add ourselves to the processor's thread list - if we're the first
	  //  we're responsible for calling the proc init task
	  {
	    // HACK: for now, don't wait - this feels like a race condition,
	    //  but the high level expects it to be this way
	    bool wait_for_init_done = false;

	    AutoHSLLock a(proc->mutex);

	    bool first = proc->all_threads.size() == 0;
	    proc->all_threads.insert(me);

	    
	    if(first) {
	      // let go of the lock while we call the init task
	      Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_INIT);
	      if(it != task_id_table.end()) {
		log_task(LEVEL_INFO, "calling processor init task: proc=%x", proc->me.id);
		proc->active_thread_count++;
		gasnet_hsl_unlock(&proc->mutex);
		(it->second)(0, 0, proc->me);
		gasnet_hsl_lock(&proc->mutex);
		proc->active_thread_count--;
		log_task(LEVEL_INFO, "finished processor init task: proc=%x", proc->me.id);
	      } else {
		log_task(LEVEL_INFO, "no processor init task: proc=%x", proc->me.id);
	      }

	      // now we can set 'init_done', and signal anybody who managed to
	      //  get themselves onto the thread list in the meantime
	      proc->init_done = true;
	      if(wait_for_init_done)
		for(std::set<Thread *>::iterator it = proc->all_threads.begin();
		    it != proc->all_threads.end();
		    it++)
		  gasnett_cond_signal(&((*it)->condvar));
	    } else {
	      // others just wait until 'init_done' becomes set
	      while(wait_for_init_done && !proc->init_done) {
		log_task(LEVEL_INFO, "waiting for processor init to complete");
		gasnett_cond_wait(&me->condvar, &proc->mutex.lock);
	      }
	    }
	  }

	  while(!proc->shutdown_requested) {
	    AutoHSLLock a(proc->mutex);

	    // pick a task, if we're allowed to (i.e. not at active thread limit)
	    assert(me->task == 0);
	    me->task = proc->select_task();

	    // didn't get one?  sleep until somebody assigns one to us
	    if(!me->task) {
	      me->state = STATE_IDLE;
	      proc->avail_threads.push_back(me);
	      log_task(LEVEL_DEBUG, "no task for thread, sleeping: proc=%x thread=%p",
		       proc->me.id, me);
	      gasnett_cond_wait(&me->condvar, &proc->mutex.lock);

	      // when we wake up, expect to have a task
	      assert(me->task != 0);
	    }
#if 0
	    // see if there's work sitting around (and we're allowed to 
	    //  start running) - if not, sleep until somebody assigns us work
	    if((proc->ready_tasks.size() > 0) &&
	       (proc->active_thread_count < proc->max_active_threads)) {
	      me->task = proc->ready_tasks.front();
	      proc->ready_tasks.pop_front();
	      proc->active_thread_count++;
	      log_task(LEVEL_SPEW, "ATC = %d", proc->active_thread_count);
	      log_task(LEVEL_DEBUG, "thread claiming ready task: proc=%x task=%p thread=%p", proc->me.id, me->task, me);
	    } else {
	      // we're idle - see if somebody else is already calling the
	      //  idle task (or if we're at the limit of active threads
	      if(proc->idle_task && !proc->in_idle_task &&
		 (proc->active_thread_count < proc->max_active_threads)) {
		proc->in_idle_task = true;
		proc->active_thread_count++;
		log_task(LEVEL_SPEW, "ATC = %d", proc->active_thread_count);
		log_task(LEVEL_SPEW, "thread calling idle task: proc=%x", proc->me.id);
		me->task = proc->idle_task;
	      } else {
		me->state = STATE_IDLE;
		proc->avail_threads.push_back(me);
		log_task(LEVEL_DEBUG, "no task for thread, sleeping: proc=%x thread=%p",
			 proc->me.id, me);
		gasnett_cond_wait(&me->condvar, &proc->mutex.lock);

		// when we wake up, expect to have a task
		assert(me->task != 0);
	      }
	    }
#endif

	    me->state = STATE_RUN;

	    // release lock while task is running
	    gasnet_hsl_unlock(&proc->mutex);
	    
	    me->task->run();

	    // retake lock, decrement active thread count
	    gasnet_hsl_lock(&proc->mutex);

	    // TODO: delete task?
	    if(me->task == proc->idle_task) {
	      log_task(LEVEL_SPEW, "thread returned from idle task: proc=%x", proc->me.id);
	      proc->in_idle_task = false;
	    }
	    me->task = 0;

	    proc->active_thread_count--;
	    log_task(LEVEL_SPEW, "ATC = %d", proc->active_thread_count);
	  }

	  {
	    AutoHSLLock a(proc->mutex);

	    // take ourselves off the list of threads - if we're the last
	    //  call a shutdown task, if one is registered
	    proc->all_threads.erase(me);
	    bool last = proc->all_threads.size() == 0;
	    
	    if(last) {
	      // let go of the lock while we call the init task
	      Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_SHUTDOWN);
	      if(it != task_id_table.end()) {
		log_task(LEVEL_INFO, "calling processor shutdown task: proc=%x", proc->me.id);
		gasnet_hsl_unlock(&proc->mutex);
		(it->second)(0, 0, proc->me);
		gasnet_hsl_lock(&proc->mutex);
		log_task(LEVEL_INFO, "finished processor shutdown task: proc=%x", proc->me.id);
	      }
	    }
	  }
	  log_task(LEVEL_DEBUG, "worker thread terminating: proc=%x", proc->me.id);
	  return 0;
	}

      public:
	LocalProcessor *proc;
	Task *task;
	State state;
	pthread_t thread;
	gasnett_cond_t condvar;
	bool instant_trigger_guard;
	Event suspend_event;
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
	  log_task(LEVEL_DEBUG, "deferred task now ready: func=%d finish=%x/%d",
		   task->func_id, 
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
	  total_threads(_total_threads),
	  active_thread_count(0), max_active_threads(_max_active_threads),
	  init_done(false), shutdown_requested(false), in_idle_task(false)
      {
	// if a processor-idle task is in the table, make a Task object for it
	Processor::TaskIDTable::iterator it = task_id_table.find(Processor::TASK_ID_PROCESSOR_IDLE);
	idle_task = ((it != task_id_table.end()) ?
  		       new Task(this, Processor::TASK_ID_PROCESSOR_IDLE, 0, 0, Event::NO_EVENT) :
		       0);
      }

      ~LocalProcessor(void)
      {
	delete idle_task;
      }

      void start_worker_threads(void)
      {
	// create worker threads - they will enqueue themselves when
	//   they're ready
	for(int i = 0; i < total_threads; i++) {
	  Thread *t = new Thread(this);
	  log_task(LEVEL_DEBUG, "creating worker thread : proc=%x thread=%p", me.id, t);
	  t->start();
	}
      }

      void add_ready_task(Task *task)
      {
	// modifications to task/thread lists require mutex
	AutoHSLLock a(mutex);

	// special case: if task->func_id is 0, that's a shutdown request
	if(task->func_id == 0) {
	  log_task(LEVEL_INFO, "shutdown request received!");
	  shutdown_requested = true;
	  return;
	}

	// do we have an available thread that can run this task right now?
	if((avail_threads.size() > 0) &&
	   (active_thread_count < max_active_threads)) {
	  Thread *thread = avail_threads.front();
	  avail_threads.pop_front();
	  active_thread_count++;
	  log_task(LEVEL_SPEW, "ATC = %d", active_thread_count);
	  log_task(LEVEL_DEBUG, "assigning new task to thread: proc=%x task=%p thread=%p", me.id, task, thread);
	  thread->set_task_and_wake(task);
	} else {
	  // no?  just stuff it on ready list
	  log_task(LEVEL_DEBUG, "no thread available for new task: proc=%x task=%p", me.id, task);
	  ready_tasks.push_back(task);
	}
      }

      // picks a task (if available/allowed) for a thread to run
      Task *select_task(void)
      {
	if(active_thread_count >= max_active_threads)
	  return 0;  // can't start anything new

	if(ready_tasks.size() > 0) {
	  Task *t = ready_tasks.front();
	  ready_tasks.pop_front();
	  active_thread_count++;
	  log_task(LEVEL_DEBUG, "ready task assigned to thread: proc=%x task=%p", me.id, t);
	  return t;
	}

	// can we give them the idle task to run?
	if(idle_task && !in_idle_task) {
	  in_idle_task = true;
	  active_thread_count++;
	  log_task(LEVEL_DEBUG, "idle task assigned to thread: proc=%x", me.id);
	  return idle_task;
	}

	// nope, nothing to do
	return 0;
      }

      // see if there are resumable threads and/or new tasks to run, respecting
      //  the available thread and runnable thread limits
      // ASSUMES LOCK IS HELD BY CALLER
      void start_some_threads(void)
      {
	// favor once-running threads that now want to resume
	while((active_thread_count < max_active_threads) &&
	      (resumable_threads.size() > 0)) {
	  Thread *t = resumable_threads.front();
	  resumable_threads.pop_front();
	  active_thread_count++;
	  log_task(LEVEL_SPEW, "ATC = %d", active_thread_count);
	  t->resume_task();
	}

	// if slots are still available, start new tasks
	while((active_thread_count < max_active_threads) &&
	      (ready_tasks.size() > 0) &&
	      (avail_threads.size() > 0)) {
	  Task *task = ready_tasks.front();
	  ready_tasks.pop_front();
	  Thread *thr = avail_threads.front();
	  avail_threads.pop_front();
	  log_task(LEVEL_DEBUG, "thread assigned ready task: proc=%x task=%p thread=%p", me.id, task, thr);
	  active_thread_count++;
	  log_task(LEVEL_SPEW, "ATC = %d", active_thread_count);
	  thr->set_task_and_wake(task);
	}

	// if we still have available threads, start the idle task (unless
	//  it's already running)
	if((active_thread_count < max_active_threads) &&
	   (avail_threads.size() > 0) &&
	   idle_task && !in_idle_task) {
	  Thread *thr = avail_threads.front();
	  avail_threads.pop_front();
	  log_task(LEVEL_DEBUG, "thread assigned idle task: proc=%x task=%p thread=%p", me.id, idle_task, thr);
	  in_idle_task = true;
	  active_thread_count++;
	  thr->set_task_and_wake(idle_task);
	}
      }

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstanceUntyped> instances_needed,
			      Event start_event, Event finish_event)
      {
	// create task object to hold args, etc.
	Task *task = new Task(this, func_id, args, arglen, finish_event);

	// early out - if the event has obviously triggered (or is NO_EVENT)
	//  don't build up continuation
	if(start_event.has_triggered()) {
	  add_ready_task(task);
	} else {
	  log_task(LEVEL_DEBUG, "deferring spawn: func=%d event=%x/%d",
		   func_id, start_event.id, start_event.gen);
	  start_event.impl()->add_waiter(start_event, new DeferredTaskSpawn(task));
	}
      }

    protected:
      int core_id;
      int total_threads, active_thread_count, max_active_threads;
      std::list<Task *> ready_tasks;
      std::list<Thread *> avail_threads;
      std::list<Thread *> resumable_threads;
      std::set<Thread *> all_threads;
      gasnet_hsl_t mutex;
      bool init_done, shutdown_requested, in_idle_task;
      Task *idle_task;
    };

    struct SpawnTaskArgs {
      Processor proc;
      Processor::TaskFuncID func_id;
      Event start_event;
      Event finish_event;
    };

    void Event::wait(void) const
    {
      if(!id) return;  // special case: never wait for NO_EVENT
      Event::Impl *e = Runtime::get_runtime()->get_event_impl(*this);

      // early out case too
      if(e->has_triggered(gen)) return;

      // figure out which thread we are - better be a local CPU thread!
      void *ptr = gasnett_threadkey_get(cur_thread);
      assert(ptr != 0);
      LocalProcessor::Thread *thr = (LocalProcessor::Thread *)ptr;
      thr->sleep_on_event(*this);
    }

    // can't be static if it's used in a template...
    void handle_spawn_task_message(SpawnTaskArgs args,
				   const void *data, size_t datalen)
    {
      Processor::Impl *p = args.proc.impl();
      log_task(LEVEL_DEBUG, "remote spawn request: proc_id=%x task_id=%d event=%x/%d",
	       args.proc.id, args.func_id, args.start_event.id, args.start_event.gen);
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
			      //std::set<RegionInstanceUntyped> instances_needed,
			      Event start_event, Event finish_event)
      {
	log_task(LEVEL_DEBUG, "spawning remote task: proc=%x task=%d start=%x/%d finish=%x/%d",
		 me.id, func_id, 
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
			   //std::set<RegionInstanceUntyped> instances_needed,
			   Event wait_on) const
    {
      Processor::Impl *p = impl();
      Event finish_event = Event::Impl::create_event();
      p->spawn_task(func_id, args, arglen, //instances_needed, 
		    wait_on, finish_event);
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

      case ID::ID_INSTANCE:
	return &(get_instance_impl(id)->lock);

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
	//printf("UNKNOWN METADATA %x\n", id.id());
	AutoHSLLock a(n->mutex); // take lock before we actually allocate
	if(!n->metadatas[index]) {
	  n->metadatas[index] = new RegionMetaDataUntyped::Impl(id.convert<RegionMetaDataUntyped>());
	} 
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
      if(id.index_l() >= mem->instances.size()) {
	// haven't seen this instance before
	Node *n = &Runtime::runtime->nodes[id.node()];
	AutoHSLLock a(n->mutex); // take lock before we actually resize

	assert(id.node() != gasnet_mynode());

	size_t old_size = mem->instances.size();
	if(id.index_l() >= old_size) {
	  // still need to grow (i.e. didn't lose the race)
	  mem->instances.resize(id.index_l() + 1);

	  // don't have region/offset info - will have to pull that when
	  //  needed
	  for(unsigned i = old_size; i <= id.index_l(); i++) 
	    mem->instances[i] = 0;
	}
      }

      if(!mem->instances[id.index_l()]) {
	// haven't seen this instance before?  create a proxy (inside a mutex)
	AutoHSLLock a(mem->mutex);

	if(!mem->instances[id.index_l()]) {
	  printf("[%d] creating proxy instance: inst=%x", gasnet_mynode(), id.id());
	  mem->instances[id.index_l()] = new RegionInstanceUntyped::Impl(id.convert<RegionInstanceUntyped>(), mem->me);
	}
      }
	  
      return mem->instances[id.index_l()];
    }

    ///////////////////////////////////////////////////
    // RegionMetaData

    /*static*/ const RegionMetaDataUntyped RegionMetaDataUntyped::NO_REGION = RegionMetaDataUntyped();

    RegionMetaDataUntyped::Impl *RegionMetaDataUntyped::impl(void) const
    {
      return Runtime::runtime->get_metadata_impl(*this);
    }

    /*static*/ RegionMetaDataUntyped RegionMetaDataUntyped::create_region_untyped(size_t num_elmts, size_t elmt_size)
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
      
      return r;
    }

    RegionAllocatorUntyped RegionMetaDataUntyped::create_allocator_untyped(Memory memory) const
    {
      ID id(memory);

      // we have to calculate the number of bytes needed in case the request
      //  goes to a remote memory
      StaticAccess<RegionMetaDataUntyped::Impl> r_data(impl());
      size_t mask_size = ElementMaskImpl::bytes_needed(0, r_data->num_elmts);

      Memory::Impl *m_impl = Runtime::runtime->get_memory_impl(memory);

      return m_impl->create_allocator(*this, 2 * mask_size);
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

      size_t inst_bytes = impl()->instance_size();

      return m_impl->create_instance(*this, inst_bytes);
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
      allocator.impl()->data.memory.impl()->destroy_allocator(allocator, true);
    }

    void RegionMetaDataUntyped::destroy_instance_untyped(RegionInstanceUntyped instance) const
    {
      instance.impl()->memory.impl()->destroy_instance(instance, true);
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
	//printf("ENABLE %p %d %d %d %x\n", raw_data, offset, start, count, impl->bits[0]);
	int pos = start - first_element;
	for(int i = 0; i < count; i++) {
	  unsigned *ptr = &(impl->bits[pos >> 5]);
	  *ptr |= (1U << (pos & 0x1f));
	  pos++;
	}
	//printf("ENABLED %p %d %d %d %x\n", raw_data, offset, start, count, impl->bits[0]);
      } else {
	//printf("ENABLE(2) %x %d %d %d\n", memory.id, offset, start, count);
	Memory::Impl *m_impl = memory.impl();

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
      }
    }

    int ElementMask::find_enabled(int count /*= 1 */)
    {
      if(raw_data != 0) {
	ElementMaskImpl *impl = (ElementMaskImpl *)raw_data;
	//printf("FIND_ENABLED %p %d %d %x\n", raw_data, first_element, count, impl->bits[0]);
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
	StaticAccess<RegionMetaDataUntyped::Impl> r_data(_region.impl());
	int num_elmts = r_data->num_elmts;
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
      StaticAccess<RegionInstanceUntyped::Impl> data(this);
      m->get_bytes(data->offset + ptr_value, dst, size);
    }

    void RegionInstanceUntyped::Impl::put_bytes(unsigned ptr_value, const void *src, size_t size)
    {
      Memory::Impl *m = Runtime::runtime->get_memory_impl(memory);
      StaticAccess<RegionInstanceUntyped::Impl> data(this);
      m->put_bytes(data->offset + ptr_value, src, size);
    }

    /*static*/ const RegionInstanceUntyped RegionInstanceUntyped::NO_INST = RegionInstanceUntyped();

    // a generic accessor just holds a pointer to the impl and passes all 
    //  requests through
    RegionInstanceAccessorUntyped<AccessorGeneric> RegionInstanceUntyped::get_accessor_untyped(void) const
    {
      return RegionInstanceAccessorUntyped<AccessorGeneric>((void *)impl());
    }

    /*static*/ void RegionInstanceUntyped::Impl::copy(RegionInstanceUntyped src, 
						      RegionInstanceUntyped target,
						      size_t bytes_to_copy,
						      Event after_copy /*= Event::NO_EVENT*/)
    {
      RegionInstanceUntyped::Impl *src_impl = src.impl();
      RegionInstanceUntyped::Impl *tgt_impl = target.impl();

      StaticAccess<RegionInstanceUntyped::Impl> src_data(src_impl);
      StaticAccess<RegionInstanceUntyped::Impl> tgt_data(tgt_impl);

      Memory::Impl *src_mem = src_impl->memory.impl();
      Memory::Impl *tgt_mem = tgt_impl->memory.impl();

      switch(src_mem->kind) {
      case Memory::Impl::MKIND_SYSMEM:
      case Memory::Impl::MKIND_ZEROCOPY:
	{
	  const void *src_ptr = src_mem->get_direct_ptr(src_data->offset, bytes_to_copy);
	  assert(src_ptr != 0);

	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

	      memcpy(tgt_ptr, src_ptr, bytes_to_copy);
	    }
	    break;

	  case Memory::Impl::MKIND_GASNET:
	    {
	      tgt_mem->put_bytes(tgt_data->offset, src_ptr, bytes_to_copy);
	    }
	    break;

	  default:
	    assert(0);
	  }
	}
	break;

      case Memory::Impl::MKIND_GASNET:
	{
	  switch(tgt_mem->kind) {
	  case Memory::Impl::MKIND_SYSMEM:
	  case Memory::Impl::MKIND_ZEROCOPY:
	    {
	      void *tgt_ptr = tgt_mem->get_direct_ptr(tgt_data->offset, bytes_to_copy);
	      assert(tgt_ptr != 0);

	      src_mem->get_bytes(src_data->offset, tgt_ptr, bytes_to_copy);
	    }
	    break;

	  case Memory::Impl::MKIND_GASNET:
	    {
	      const unsigned BLOCK_SIZE = 4096;
	      unsigned char temp_block[BLOCK_SIZE];

	      size_t bytes_copied = 0;
	      while(bytes_copied < bytes_to_copy) {
		size_t chunk_size = bytes_to_copy - bytes_copied;
		if(chunk_size > BLOCK_SIZE) chunk_size = BLOCK_SIZE;

		src_mem->get_bytes(src_data->offset + bytes_copied, temp_block, chunk_size);
		tgt_mem->put_bytes(tgt_data->offset + bytes_copied, temp_block, chunk_size);
		bytes_copied += chunk_size;
	      }
	    }
	    break;

	  default:
	    assert(0);
	  }
	}
	break;

      default:
	assert(0);
      }

      // don't forget to release the locks on the instances!
      src_impl->lock.unlock();
      if(tgt_impl != src_impl)
	tgt_impl->lock.unlock();

      if(after_copy.exists())
	after_copy.impl()->trigger(after_copy.gen, true);
    }

    class DeferredCopy : public Event::Impl::EventWaiter {
    public:
      DeferredCopy(RegionInstanceUntyped _src, RegionInstanceUntyped _target,
		   size_t _bytes_to_copy, Event _after_copy)
	: src(_src), target(_target), 
	  bytes_to_copy(_bytes_to_copy), after_copy(_after_copy) {}

      virtual void event_triggered(void)
      {
	RegionInstanceUntyped::Impl::copy(src, target, bytes_to_copy, after_copy);
      }

    protected:
      RegionInstanceUntyped src, target;
      size_t bytes_to_copy;
      Event after_copy;
    };

    Event RegionInstanceUntyped::copy_to_untyped(RegionInstanceUntyped target, 
						 Event wait_on /*= Event::NO_EVENT*/)
    {
      RegionInstanceUntyped::Impl *src_impl = impl();
      RegionInstanceUntyped::Impl *dst_impl = target.impl();
      Memory::Impl *src_mem = src_impl->memory.impl();
      Memory::Impl *dst_mem = dst_impl->memory.impl();

      printf("COPY %x (%d) -> %x (%d)\n", id, src_mem->kind, target.id, dst_mem->kind);

      // first check - if either or both memories are remote, we're going to
      //  need to find somebody else to do this copy
      if((src_mem->kind == Memory::Impl::MKIND_REMOTE) ||
	 (dst_mem->kind == Memory::Impl::MKIND_REMOTE)) {
	assert(0);
      }

      size_t bytes_to_copy = StaticAccess<RegionInstanceUntyped::Impl>(src_impl)->region.impl()->instance_size();

      // ok - we're going to do the copy locally, but we need locks on the 
      //  two instances, and those locks may need to be deferred
      Event lock_event = src_impl->lock.lock(1, false, wait_on);
      if(dst_impl != src_impl) {
	Event dst_lock_event = dst_impl->lock.lock(1, false, wait_on);
	lock_event = Event::merge_events(lock_event, dst_lock_event);
      }
      // now do we have to wait?
      if(!lock_event.has_triggered()) {
	Event after_copy = Event::Impl::create_event();
	lock_event.impl()->add_waiter(lock_event,
				      new DeferredCopy(*this, target,
						       bytes_to_copy, 
						       after_copy));
	return after_copy;
      }

      // we can do the copy immediately here
      RegionInstanceUntyped::Impl::copy(*this, target, bytes_to_copy);
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
	//usleep(10000);
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

      Logger::init(*argc, (const char **)*argv);

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
      hcount += CreateAllocatorMessage::add_handler_entries(&handlers[hcount]);
      hcount += CreateInstanceMessage::add_handler_entries(&handlers[hcount]);
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
      std::set<LocalProcessor *> local_cpu_procs;

      for(unsigned i = 0; i < num_local_procs; i++) {
	LocalProcessor *lp = new LocalProcessor(ID(ID::ID_PROCESSOR, gasnet_mynode(), i).convert<Processor>(), i, 1, 1);
	n->processors[i] = lp;
	local_cpu_procs.insert(lp);
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

      // now that we've got the machine description all set up, we can start
      //  the worker threads for local processors, which'll probably ask the
      //  high-level runtime to set itself up
      for(std::set<LocalProcessor *>::iterator it = local_cpu_procs.begin();
	  it != local_cpu_procs.end();
	  it++)
	(*it)->start_worker_threads();
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
      for(int i = 0; i < 1000; i++) {
	fflush(stdout);
	sleep(1);
      }
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
