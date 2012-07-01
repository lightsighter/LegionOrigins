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
#include <queue>
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
      AutoHSLLock(gasnet_hsl_t &mutex) : mutexp(&mutex), held(true)
      { 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK IN %p", mutexp);
	//printf("[%d] MUTEX LOCK IN %p\n", gasnet_mynode(), mutexp);
	gasnet_hsl_lock(mutexp); 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK HELD %p", mutexp);
	//printf("[%d] MUTEX LOCK HELD %p\n", gasnet_mynode(), mutexp);
      }
      AutoHSLLock(gasnet_hsl_t *_mutexp) : mutexp(_mutexp), held(true)
      { 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK IN %p", mutexp);
	//printf("[%d] MUTEX LOCK IN %p\n", gasnet_mynode(), mutexp);
	gasnet_hsl_lock(mutexp); 
	log_mutex(LEVEL_SPEW, "MUTEX LOCK HELD %p", mutexp);
	//printf("[%d] MUTEX LOCK HELD %p\n", gasnet_mynode(), mutexp);
      }
      ~AutoHSLLock(void) 
      {
	if(held)
	  gasnet_hsl_unlock(mutexp);
	log_mutex(LEVEL_SPEW, "MUTEX LOCK OUT %p", mutexp);
	//printf("[%d] MUTEX LOCK OUT %p\n", gasnet_mynode(), mutexp);
      }
      void release(void)
      {
	assert(held);
	gasnet_hsl_unlock(mutexp);
	held = false;
      }
      void reacquire(void)
      {
	assert(!held);
	gasnet_hsl_lock(mutexp);
	held = true;
      }
    protected:
      gasnet_hsl_t *mutexp;
      bool held;
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
      typedef unsigned long long uint64;
      uint64_t dummy;
      uint64_t bits[0];

      static size_t bytes_needed(off_t offset, off_t count)
      {
	size_t need = sizeof(ElementMaskImpl) + (((count + 63) >> 6) << 3);
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

      enum { MODE_EXCL = 0, ZERO_COUNT = 0x11223344 };

      gasnet_hsl_t *mutex; // controls which local thread has access to internal data (not runtime-visible lock)

      // bitmasks of which remote nodes are waiting on a lock (or sharing it)
      uint64_t remote_waiter_mask, remote_sharer_mask;
      //std::list<LockWaiter *> local_waiters; // set of local threads that are waiting on lock
      std::map<unsigned, std::deque<Event> > local_waiters;
      bool requested; // do we have a request for the lock in flight?

      // local data protected by lock
      void *local_data;
      size_t local_data_size;

      static gasnet_hsl_t freelist_mutex;
      static Lock::Impl *first_free;
      Lock::Impl *next_free;

      Event lock(unsigned new_mode, bool exclusive,
		 Event after_lock = Event::NO_EVENT);

      bool select_local_waiters(std::deque<Event>& to_wake);

      void unlock(void);

      bool is_locked(unsigned check_mode, bool excl_ok);

      void release_lock(void);
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

    class UtilityProcessor;

    class Processor::Impl {
    public:
      Impl(Processor _me, Processor::Kind _kind, Processor _util = Processor::NO_PROC)
	: me(_me), kind(_kind), util(_util), util_proc(0), run_counter(0) {}

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

      void set_utility_processor(UtilityProcessor *_util_proc);

      virtual void enable_idle_task(void) { assert(0); }
      virtual void disable_idle_task(void) { assert(0); }
      virtual bool is_idle_task_enabled(void) { return(false); }

    public:
      Processor me;
      Processor::Kind kind;
      Processor util;
      UtilityProcessor *util_proc;
      Atomic<int> *run_counter;
    };
    
    class PreemptableThread {
    public:
      PreemptableThread(void) {}
      virtual ~PreemptableThread(void) {}

      void start_thread(void);

      static bool preemptable_sleep(Event wait_for, bool block = false);

    protected:
      static void *thread_entry(void *data);

      virtual void thread_main(void) = 0;

      virtual void sleep_on_event(Event wait_for, bool block = false) = 0;

      pthread_t thread;
    };

    class UtilityProcessor : public Processor::Impl {
    public:
      UtilityProcessor(Processor _me, int _num_worker_threads = 1);
      virtual ~UtilityProcessor(void);

      void start_worker_threads(void);

      virtual void spawn_task(Processor::TaskFuncID func_id,
			      const void *args, size_t arglen,
			      //std::set<RegionInstanceUntyped> instances_needed,
			      Event start_event, Event finish_event);

      void request_shutdown(void);

      void enable_idle_task(Processor::Impl *proc);
      void disable_idle_task(Processor::Impl *proc);

      void wait_for_shutdown(void);

      class UtilityThread;
      class UtilityTask;

    protected:
      //friend class UtilityThread;
      //friend class UtilityTask;

      void enqueue_runnable_task(UtilityTask *task);

      int num_worker_threads;
      bool shutdown_requested;

      gasnet_hsl_t mutex;
      gasnett_cond_t condvar;

      UtilityTask *idle_task;

      std::set<UtilityThread *> threads;
      std::queue<UtilityTask *> tasks;
      std::set<Processor::Impl *> idle_procs;
      std::set<Processor::Impl *> procs_in_idle_task;
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

    Impl(Memory _me, size_t _size, MemoryKind _kind, size_t _alignment)
      : me(_me), size(_size), kind(_kind), alignment(_alignment)
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
						  size_t bytes_needed,
						  off_t adjust);
      RegionInstanceUntyped create_instance_local(RegionMetaDataUntyped r,
						  size_t bytes_needed,
						  off_t adjust,
						  ReductionOpID redopid);

      RegionAllocatorUntyped create_allocator_remote(RegionMetaDataUntyped r,
						     size_t bytes_needed);
      RegionInstanceUntyped create_instance_remote(RegionMetaDataUntyped r,
						   size_t bytes_needed,
						   off_t adjust);
      RegionInstanceUntyped create_instance_remote(RegionMetaDataUntyped r,
						   size_t bytes_needed,
						   off_t adjust,
						   ReductionOpID redopid);

      virtual RegionAllocatorUntyped create_allocator(RegionMetaDataUntyped r,
						      size_t bytes_needed) = 0;
      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed,
						    off_t adjust) = 0;
      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed,
						    off_t adjust,
						    ReductionOpID redopid) = 0;

      void destroy_allocator(RegionAllocatorUntyped a, bool local_destroy);

      void destroy_instance_local(RegionInstanceUntyped i, bool local_destroy);
      void destroy_instance_remote(RegionInstanceUntyped i, bool local_destroy);

      virtual void destroy_instance(RegionInstanceUntyped i, 
				    bool local_destroy) = 0;

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
      size_t alignment;
      gasnet_hsl_t mutex; // protection for resizing vectors
      std::vector<RegionAllocatorUntyped::Impl *> allocators;
      std::vector<RegionInstanceUntyped::Impl *> instances;
      std::map<off_t, off_t> free_blocks;
    };

    class GASNetMemory : public Memory::Impl {
    public:
      static const size_t MEMORY_STRIDE = 1024;

      GASNetMemory(Memory _me, size_t size_per_node);

      virtual ~GASNetMemory(void);

      virtual RegionAllocatorUntyped create_allocator(RegionMetaDataUntyped r,
						      size_t bytes_needed);

      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed,
						    off_t adjust);

      virtual RegionInstanceUntyped create_instance(RegionMetaDataUntyped r,
						    size_t bytes_needed,
						    off_t adjust,
						    ReductionOpID redopid);

      virtual void destroy_instance(RegionInstanceUntyped i, 
				    bool local_destroy);

      virtual off_t alloc_bytes(size_t size);

      virtual void free_bytes(off_t offset, size_t size);

      virtual void get_bytes(off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);

      void get_batch(size_t batch_size,
		     const off_t *offsets, void * const *dsts, 
		     const size_t *sizes);

      void put_batch(size_t batch_size,
		     const off_t *offsets, const void * const *srcs, 
		     const size_t *sizes);

    protected:
      int num_nodes;
      off_t memory_stride;
      gasnet_seginfo_t *seginfos;
      //std::map<off_t, off_t> free_blocks;
    };

    class RegionInstanceUntyped::Impl {
    public:
      Impl(RegionInstanceUntyped _me, RegionMetaDataUntyped _region, Memory _memory, off_t _offset, size_t _size, off_t _adjust);

      Impl(RegionInstanceUntyped _me, RegionMetaDataUntyped _region, Memory _memory, off_t _offset, size_t _size, off_t _adjust, ReductionOpID _redopid);

      // when we auto-create a remote instance, we don't know region/offset
      Impl(RegionInstanceUntyped _me, Memory _memory);

      ~Impl(void);

#ifdef POINTER_CHECKS
      void verify_access(unsigned ptr);
      const ElementMask& get_element_mask(void);
#endif
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
	off_t alloc_offset, access_offset;
	size_t size;
	size_t first_elmt, last_elmt;
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
      Event::Impl *next_free;
      static Event::Impl *first_free;
      static gasnet_hsl_t freelist_mutex;

      gasnet_hsl_t *mutex; // controls which local thread has access to internal data (not runtime-visible event)

      uint64_t remote_waiters; // bitmask of which remote nodes are waiting on the event
      std::map<Event::gen_t, std::vector<EventWaiter *> > local_waiters; // set of local threads that are waiting on event (keyed by generation)
    };

    class RegionMetaDataUntyped::Impl {
    public:
      Impl(RegionMetaDataUntyped _me, RegionMetaDataUntyped _parent,
	   size_t _num_elmts, size_t _elmt_size,
	   const ElementMask *_initial_valid_mask = 0, bool _frozen = false);

      // this version is called when we create a proxy for a remote region
      Impl(RegionMetaDataUntyped _me);

      ~Impl(void);

      bool is_parent_of(RegionMetaDataUntyped other);

      size_t instance_size(const ReductionOpUntyped *redop = 0);

      off_t instance_adjust(const ReductionOpUntyped *redop = 0);

      Event request_valid_mask(void);

      RegionMetaDataUntyped me;
      Lock::Impl lock;

      struct StaticData {
	bool valid;
	RegionMetaDataUntyped parent;
	bool frozen;
	size_t num_elmts, elmt_size;
        size_t first_elmt, last_elmt;
      };
      struct CoherentData : public StaticData {
	unsigned valid_mask_owners;
	int avail_mask_owner;
      };

      CoherentData locked_data;
      gasnet_hsl_t valid_mask_mutex;
      ElementMask *valid_mask;
      int valid_mask_count;
      bool valid_mask_complete;
      Event valid_mask_event;
      int valid_mask_first, valid_mask_last;
      bool valid_mask_contig;
      ElementMask *avail_mask;
    };

  }; // namespace LowLevel
}; // namespace RegionRuntime

#endif
