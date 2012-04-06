#ifndef LOWLEVEL_IMPL_H
#define LOWLEVEL_IMPL_H

#include "lowlevel.h"

#include <assert.h>

#ifndef NO_INCLUDE_GASNET
#define GASNET_PAR
#include <gasnet.h>

#define GASNETT_THREAD_SAFE
#include <gasnet_tools.h>

#include "activemsg.h"
#endif

#ifdef CHECK_REENTRANT_MESSAGES
GASNETT_THREADKEY_DEFINE(in_handler);
#endif
GASNETT_THREADKEY_DECLARE(cur_thread);

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
    extern Logger::Category log_mutex;

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
      Node(void);

      gasnet_hsl_t mutex;  // used to cover resizing activities on vectors below
      std::vector<Event::Impl> events;
      size_t num_events;
      std::vector<Lock::Impl> locks;
      size_t num_locks;
      std::vector<Memory::Impl *> memories;
      std::vector<Processor::Impl *> processors;
      std::vector<RegionMetaDataUntyped::Impl *> metadatas;
    };

    template <class T>
    class Atomic {
    public:
      Atomic(T _value) : value(_value)
      {
	gasnet_hsl_init(&mutex);
      }

      T get(void) const { return value; }

      void decrement(void)
      {
	AutoHSLLock a(mutex);
	value--;
      }

    protected:
      T value;
      gasnet_hsl_t mutex;
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

    struct ElementMaskImpl {
      //int count, offset;
      int dummy;
      unsigned bits[0];

      static size_t bytes_needed(off_t offset, off_t count)
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
	    thing_with_data->lock.lock(1, false).wait(true);// TODO: must this be blocking?
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

    template <class T>
    class ExclusiveAccess {
    public:
      typedef typename T::CoherentData CoherentData;

      // if already_held, just check that it's held (if in debug mode)
      ExclusiveAccess(T* thing_with_data, bool already_held = false)
	: data(&thing_with_data->locked_data), lock(&thing_with_data->lock)
      {
	if(already_held) {
	  assert(lock->is_locked(0, true));
	} else {
	  lock->lock(0, true).wait();
	}
      }

      ~ExclusiveAccess(void)
      {
	lock->unlock();
      }

      CoherentData *operator->(void) { return data; }

    protected:
      CoherentData *data;
      Lock::Impl *lock;
    };

    extern Processor::TaskIDTable task_id_table;

    class Processor::Impl {
    public:
      Impl(Processor _me, Processor::Kind _kind)
	: me(_me), kind(_kind), run_counter(0) {}

      void run(Atomic<int> *_run_counter)
      {
	run_counter = _run_counter;
      }

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstanceUntyped> instances_needed,
			      Event start_event, Event finish_event) = 0;

      void finished(void)
      {
	if(run_counter)
	  run_counter->decrement();
      }

      virtual void enable_idle_task(void) { assert(0); }
      virtual void disable_idle_task(void) { assert(0); }

    public:
      Processor me;
      Processor::Kind kind;
      Atomic<int> *run_counter;
    };

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
      RegionInstanceUntyped create_instance_local(RegionMetaDataUntyped r,
						  size_t bytes_needed,
						  ReductionOpID redopid);

      RegionAllocatorUntyped create_allocator_remote(RegionMetaDataUntyped r,
						     size_t bytes_needed);
      RegionInstanceUntyped create_instance_remote(RegionMetaDataUntyped r,
						   size_t bytes_needed);
      RegionInstanceUntyped create_instance_remote(RegionMetaDataUntyped r,
						   size_t bytes_needed,
						   ReductionOpID redopid);

      virtual RegionAllocatorUntyped create_allocator(RegionMetaDataUntyped r,
						      size_t bytes_needed) = 0;
      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed) = 0;
      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed,
						    ReductionOpID redopid) = 0;

      void destroy_allocator(RegionAllocatorUntyped a, bool local_destroy);
      void destroy_instance(RegionInstanceUntyped i, bool local_destroy);

      off_t alloc_bytes_local(size_t size);
      void free_bytes_local(off_t offset, size_t size);

      off_t alloc_bytes_remote(size_t size);
      void free_bytes_remote(off_t offset, size_t size);

      virtual off_t alloc_bytes(size_t size) = 0;
      virtual void free_bytes(off_t offset, size_t size) = 0;

      virtual void get_bytes(off_t offset, void *dst, size_t size) = 0;
      virtual void put_bytes(off_t offset, const void *src, size_t size) = 0;

      virtual void *get_direct_ptr(off_t offset, size_t size) = 0;

    public:
      Memory me;
      size_t size;
      MemoryKind kind;
      gasnet_hsl_t mutex; // protection for resizing vectors
      std::vector<RegionAllocatorUntyped::Impl *> allocators;
      std::vector<RegionInstanceUntyped::Impl *> instances;
      std::map<off_t, off_t> free_blocks;
    };

    class RegionInstanceUntyped::Impl {
    public:
      Impl(RegionInstanceUntyped _me, RegionMetaDataUntyped _region, Memory _memory, off_t _offset);

      Impl(RegionInstanceUntyped _me, RegionMetaDataUntyped _region, Memory _memory, off_t _offset, ReductionOpID _redopid);

      // when we auto-create a remote instance, we don't know region/offset
      Impl(RegionInstanceUntyped _me, Memory _memory);

      ~Impl(void);

      void get_bytes(off_t ptr_value, void *dst, size_t size);
      void put_bytes(off_t ptr_value, const void *src, size_t size);

      static Event copy(RegionInstanceUntyped src, 
			RegionInstanceUntyped target,
			RegionMetaDataUntyped region,
			size_t elmt_size,
			size_t bytes_to_copy,
			Event after_copy = Event::NO_EVENT);

    public: //protected:
      friend class RegionInstanceUntyped;

      RegionInstanceUntyped me;
      Memory memory;

      struct StaticData {
	bool valid;
	RegionMetaDataUntyped region;
	off_t offset;
	bool is_reduction;
	ReductionOpID redopid;
      } locked_data;

      Lock::Impl lock;
    };

    class Event::Impl {
    public:
      Impl(void);

      void init(Event _me, unsigned _init_owner);

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
      void trigger(Event::gen_t gen_triggered, int trigger_node);

      class EventWaiter {
      public:
	virtual void event_triggered(void) = 0;
	virtual void print_info(void) = 0;
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

  }; // namespace LowLevel
}; // namespace RegionRuntime

#endif
