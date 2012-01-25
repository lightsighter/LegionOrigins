#ifndef RUNTIME_LOWLEVEL_H
#define RUNTIME_LOWLEVEL_H

#include <string>
#include <vector>
#include <set>
#include <map>
#include <cstdarg>

#include "common.h"
#include "utilities.h"

namespace RegionRuntime {
  namespace LowLevel {
    // forward class declarations because these things all refer to each other
    class Event;
    class UserEvent;
    class Lock;
    class Memory;
    class Processor;
    class RegionMetaDataUntyped;
    class RegionAllocatorUntyped;
    class RegionInstanceUntyped;
    template <class T> class RegionMetaData;
    template <class T> class RegionAllocator;
    template <class T> class RegionInstance;

    class Event {
    public:
      typedef unsigned id_t;
      typedef unsigned gen_t;
#if 0
      typedef unsigned long long fused_t;
      union {
	fused_t fused;
	struct {
	  id_t id;
	  gen_t gen;
	}
      };
#endif
      id_t id;
      gen_t gen;
      bool operator<(const Event& rhs) const { return id < rhs.id; }
      bool operator==(const Event& rhs) const { return id == rhs.id; }
      bool operator!=(const Event& rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const Event NO_EVENT;

      bool exists(void) const { return id != 0; }

      // test whether an event has triggered without waiting
      bool has_triggered(void) const;

      // causes calling thread to block until event has occurred
      void wait(void) const;

      // creates an event that won't trigger until all input events have
      static Event merge_events(const std::set<Event>& wait_for);
      static Event merge_events(Event ev1, Event ev2,
				Event ev3 = NO_EVENT, Event ev4 = NO_EVENT,
				Event ev5 = NO_EVENT, Event ev6 = NO_EVENT);
    };

    // A user level event has all the properties of event, except
    // it can be triggered by the user.  This prevents users from
    // triggering arbitrary events without doing something like
    // an unsafe cast.
    class UserEvent : public Event {
    public:
      static UserEvent create_user_event(void);
      void trigger(void) const;
    };

    class Lock {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const Lock& rhs) const { return id < rhs.id; }
      bool operator==(const Lock& rhs) const { return id == rhs.id; }
      bool operator!=(const Lock& rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const Lock NO_LOCK;

      bool exists(void) const { return id != 0; }

      // requests ownership (either exclusive or shared) of the lock with a 
      //   specified mode - returns an event that will trigger when the lock
      //   is granted
      Event lock(unsigned mode = 0, bool exclusive = true, Event wait_on = Event::NO_EVENT) const;
      // releases a held lock - release can be deferred until an event triggers
      void unlock(Event wait_on = Event::NO_EVENT) const;

      // Create a new lock, destroy an existing lock
      static Lock create_lock(void);
      void destroy_lock();
  };

    class Processor {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const Processor& rhs) const { return id < rhs.id; }
      bool operator==(const Processor& rhs) const { return id == rhs.id; }
      bool operator!=(const Processor& rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const Processor NO_PROC;

      bool exists(void) const { return id != 0; }

      Processor get_utility_processor(void) const { return *this; }

      void enable_idle_task(void) { }
      void disable_idle_task(void) { }

      typedef unsigned TaskFuncID;
      typedef void (*TaskFuncPtr)(const void *args, size_t arglen, Processor proc);
      typedef std::map<TaskFuncID, TaskFuncPtr> TaskIDTable;

      // Different Processor types
      enum Kind {
	TOC_PROC, // Throughput core
	LOC_PROC, // Latency core
      };


      // special task IDs
      enum {
        // Save ID 0 for the force shutdown function
	TASK_ID_PROCESSOR_INIT     = 1,
	TASK_ID_PROCESSOR_SHUTDOWN = 2,
	TASK_ID_PROCESSOR_IDLE     = 3, // typically used for high-level scheduler
	TASK_ID_FIRST_AVAILABLE    = 4,
      };

      Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
		  //std::set<RegionInstanceUntyped> instances_needed,
		  Event wait_on = Event::NO_EVENT) const;
    };

    class Memory {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const Memory &rhs) const { return id < rhs.id; }
      bool operator==(const Memory &rhs) const { return id == rhs.id; }
      bool operator!=(const Memory &rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const Memory NO_MEMORY;

      bool exists(void) const { return id != 0; }
    };

    class ElementMask {
    public:
      ElementMask(void);
      ElementMask(int num_elements, int first_element = 0);
      ElementMask(const ElementMask &copy_from, int num_elements = -1, int first_element = 0);

      void init(int _first_element, int _num_elements, Memory _memory, off_t _offset);

      int get_num_elmts(void) const { return num_elements; }

      void enable(int start, int count = 1);
      void disable(int start, int count = 1);

      int find_enabled(int count = 1);
      int find_disabled(int count = 1);
      
      bool is_set(int ptr) const;
      // union/intersect/subtract?

      int first_enabled(void) const;
      int last_enabled(void) const;

      class Enumerator {
      public:
	Enumerator(const ElementMask& _mask, int _start, int _polarity);
	~Enumerator(void);

	bool get_next(int &position, int &length);

      protected:
	const ElementMask& mask;
	int pos;
	int polarity;
      };

      Enumerator *enumerate_enabled(int start = 0) const;
      Enumerator *enumerate_disabled(int start = 0) const;

      size_t raw_size(void) const;
      const void *get_raw(void) const;
      void set_raw(const void *data);

    protected:
      friend class Enumerator;
      int first_element;
      int num_elements;
      Memory memory;
      off_t offset;
      void *raw_data;
      int first_enabled_elmt, last_enabled_elmt;
    };

    class RegionMetaDataUntyped {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const RegionMetaDataUntyped &rhs) const { return id < rhs.id; }
      bool operator==(const RegionMetaDataUntyped &rhs) const { return id == rhs.id; }
      bool operator!=(const RegionMetaDataUntyped &rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const RegionMetaDataUntyped NO_REGION;

      bool exists(void) const { return id != 0; }

      static RegionMetaDataUntyped create_region_untyped(size_t num_elmts, size_t elmt_size);
      static RegionMetaDataUntyped create_region_untyped(RegionMetaDataUntyped parent, const ElementMask &mask);

      RegionAllocatorUntyped create_allocator_untyped(Memory memory) const;
      RegionInstanceUntyped create_instance_untyped(Memory memory) const;

      void destroy_region_untyped(void) const;
      void destroy_allocator_untyped(RegionAllocatorUntyped allocator) const;
      void destroy_instance_untyped(RegionInstanceUntyped instance) const;

      const ElementMask &get_valid_mask(void);
    };

    class RegionAllocatorUntyped {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const RegionAllocatorUntyped &rhs) const { return id < rhs.id; }
      bool operator==(const RegionAllocatorUntyped &rhs) const { return id == rhs.id; }
      bool operator!=(const RegionAllocatorUntyped &rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const RegionAllocatorUntyped NO_ALLOC;

      bool exists(void) const { return id != 0; }

    protected:
      unsigned alloc_untyped(unsigned count = 1) const;
      void free_untyped(unsigned ptr, unsigned count = 1) const;
    };

    enum AccessorType { AccessorGeneric, AccessorArray, AccessorGPU };

    template <AccessorType AT> class RegionInstanceAccessorUntyped;

    template <> class RegionInstanceAccessorUntyped<AccessorGeneric> {
    public:
      explicit RegionInstanceAccessorUntyped(void *_internal_data)
	: internal_data(_internal_data) {}

      // Need copy constructors so we can move things around
      RegionInstanceAccessorUntyped(const RegionInstanceAccessorUntyped<AccessorGeneric> &old)
      { internal_data = old.internal_data; }

      void *internal_data;

      void get_untyped(unsigned ptr_value, void *dst, size_t size) const;
      void put_untyped(unsigned ptr_value, const void *src, size_t size) const;

      template <class T>
      T read(ptr_t<T> ptr) const
	{ T val; get_untyped(ptr.value*sizeof(T), &val, sizeof(T)); return val; }

      template <class T>
      void write(ptr_t<T> ptr, T newval) const
	{ put_untyped(ptr.value*sizeof(T), &newval, sizeof(T)); }

      template <class REDOP, class T, class RHS>
      void reduce(ptr_t<T> ptr, RHS newval) const 
	{ T val; 
	  get_untyped(ptr.value*sizeof(T), &val, sizeof(T));
	  REDOP::apply(&val, newval); 
	  put_untyped(ptr.value*sizeof(T), &val, sizeof(T)); }

      template <AccessorType AT2>
      bool can_convert(void) const;

      template <AccessorType AT2>
      RegionInstanceAccessorUntyped<AT2> convert(void) const;
    };

    template <> class RegionInstanceAccessorUntyped<AccessorArray> {
    public:
      explicit RegionInstanceAccessorUntyped(void *_array_base)
	: array_base(_array_base) {}

      // Need copy constructors so we can move things around
      RegionInstanceAccessorUntyped(const RegionInstanceAccessorUntyped<AccessorArray> &old)
      { array_base = old.array_base; }

      void *array_base;

      template <class T>
      T read(ptr_t<T> ptr) const { return ((T*)array_base)[ptr.value]; }

      template <class T>
      void write(ptr_t<T> ptr, T newval) const { ((T*)array_base)[ptr.value] = newval; }

      template <class REDOP, class T, class RHS>
      void reduce(ptr_t<T> ptr, RHS newval) const { REDOP::apply(((T*)array_base)[ptr.value], newval); }
    };

    // only nvcc understands this
    template <> class RegionInstanceAccessorUntyped<AccessorGPU> {
    public:
      explicit RegionInstanceAccessorUntyped(void *_array_base)
	: array_base(_array_base) {}

      // Need copy constructors so we can move things around
      RegionInstanceAccessorUntyped(const RegionInstanceAccessorUntyped<AccessorArray> &old)
      { array_base = old.array_base; }

      void *array_base;

#ifdef __CUDACC__
      template <class T>
      __device__ T read(ptr_t<T> ptr) const { return ((T*)array_base)[ptr.value]; }

      template <class T>
      __device__ void write(ptr_t<T> ptr, T newval) const { ((T*)array_base)[ptr.value] = newval; }

      template <class REDOP, class T, class RHS>
      __device__ void reduce(ptr_t<T> ptr, RHS newval) const { REDOP::apply(&((T*)array_base)[ptr.value], newval); }
#endif
    };

    template <class ET, AccessorType AT = AccessorGeneric>
    class RegionInstanceAccessor {
    public:
      RegionInstanceAccessor(const RegionInstanceAccessorUntyped<AT> &_ria) : ria(_ria) {}

      RegionInstanceAccessorUntyped<AT> ria;

      ET read(ptr_t<ET> ptr) const { return ria.read(ptr); }
      void write(ptr_t<ET> ptr, ET newval) const { ria.write(ptr, newval); }

      template <class REDOP, class RHS>
      void reduce(ptr_t<ET> ptr, RHS newval) const { ria.template reduce<REDOP>(ptr, newval); }

      template <AccessorType AT2>
      bool can_convert(void) const { return ria.can_convert<AT2>(); }

      template <AccessorType AT2>
      RegionInstanceAccessor<ET,AT2> convert(void) const
      { return RegionInstanceAccessor<ET,AT2>(ria.convert<AT2>()); }
    };

#ifdef __CUDACC__
    template <class ET>
    class RegionInstanceAccessor<ET,AccessorGPU> {
    public:
      __device__ RegionInstanceAccessor(const RegionInstanceAccessorUntyped<AccessorGPU> &_ria) : ria(_ria) {}

      RegionInstanceAccessorUntyped<AccessorGPU> ria;

      __device__ ET read(ptr_t<ET> ptr) const { return ria.read(ptr); }
      __device__ void write(ptr_t<ET> ptr, ET newval) const { ria.write(ptr, newval); }

      //template <class REDOP, class RHS>
      //void reduce(ptr_t<ET> ptr, RHS newval) const { ria.template reduce<REDOP>(ptr, newval); }
    };
#endif

    class RegionInstanceUntyped {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const RegionInstanceUntyped &rhs) const { return id < rhs.id; }
      bool operator==(const RegionInstanceUntyped &rhs) const { return id == rhs.id; }
      bool operator!=(const RegionInstanceUntyped &rhs) const { return id != rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const RegionInstanceUntyped NO_INST;

      bool exists(void) const { return id != 0; }

      RegionInstanceAccessorUntyped<AccessorGeneric> get_accessor_untyped(void) const;

      Event copy_to_untyped(RegionInstanceUntyped target, Event wait_on = Event::NO_EVENT);
      Event copy_to_untyped(RegionInstanceUntyped target, const ElementMask &mask, Event wait_on = Event::NO_EVENT);
    };

    template <class T>
    class RegionMetaData : public RegionMetaDataUntyped {
    public:
      RegionMetaData(void) {}
      RegionMetaData(const RegionMetaData<T>& copy_from)
	: RegionMetaDataUntyped(copy_from) {}

      // operator to re-introduce element type - make sure you're right!
      explicit RegionMetaData(const RegionMetaDataUntyped& copy_from)
	: RegionMetaDataUntyped(copy_from) {}

      static RegionMetaData<T> create_region(size_t _num_elems) {
	return RegionMetaData<T>(create_region_untyped(_num_elems,sizeof(T)));
      }

      RegionAllocator<T> create_allocator(Memory memory) {
	return RegionAllocator<T>(create_allocator_untyped(memory));
      }
	  
      RegionInstance<T> create_instance(Memory memory) {
	return RegionInstance<T>(create_instance_untyped(memory));
      }

      void destroy_region() {
        destroy_region_untyped();
      }

      void destroy_allocator(RegionAllocator<T> allocator) {
        destroy_allocator_untyped(allocator);
      }

      void destroy_instance(RegionInstance<T> instance) {
        destroy_instance_untyped(instance);
      }
    };

    template <class T>
    class RegionAllocator : public RegionAllocatorUntyped {
    public:
      // operator to re-introduce element type - make sure you're right!
      explicit RegionAllocator(const RegionAllocatorUntyped& copy_from)
	: RegionAllocatorUntyped(copy_from) {}
      
      ptr_t<T> alloc(void) 
      { 
	ptr_t<T> ptr = { alloc_untyped(1) };
	return ptr; 
      }
      void free(ptr_t<T> ptr) { free_untyped(ptr.value); }
    };

    template <class T>
    class RegionInstance : public RegionInstanceUntyped {
    public:
      RegionInstance(void) : RegionInstanceUntyped(NO_INST) {}

      // operator to re-introduce element type - make sure you're right!
      explicit RegionInstance(const RegionInstanceUntyped& copy_from)
	: RegionInstanceUntyped(copy_from) {}

      // the instance doesn't have read/write/reduce methods of its own -
      //  instead, we can hand out an "accessor" object that has those methods
      //  this lets us specialize for the just-an-array-dereference case
      RegionInstanceAccessor<T,AccessorGeneric> get_accessor(void) const
      { return RegionInstanceAccessor<T,AccessorGeneric>(get_accessor_untyped()); }

      Event copy_to(RegionInstance<T> target, Event wait_on = Event::NO_EVENT)
      { return copy_to_untyped(RegionInstanceUntyped(target), wait_on); }

      Event copy_to(RegionInstance<T> target, const ElementMask& mask, Event wait_on = Event::NO_EVENT)
      { return copy_to_untyped(RegionInstanceUntyped(target), mask, wait_on); }
    };

    class Machine {
    public:
      Machine(int *argc, char ***argv,
	      const Processor::TaskIDTable &task_table,
	      bool cps_style = false, Processor::TaskFuncID init_id = 0);
      ~Machine(void);

      // there are three potentially interesting ways to start the initial
      // tasks:
      enum RunStyle {
	ONE_TASK_ONLY,  // a single task on a single node of the machine
	ONE_TASK_PER_NODE, // one task running on one proc of each node
	ONE_TASK_PER_PROC, // a task for every processor in the machine
      };

      void run(Processor::TaskFuncID task_id = 0, RunStyle style = ONE_TASK_ONLY,
	       const void *args = 0, size_t arglen = 0);

    public:
      const std::set<Memory>&    get_all_memories(void) const { return memories; }
      const std::set<Processor>& get_all_processors(void) const { return procs; }
      // Return the set of memories visible from a processor
      const std::set<Memory>&    get_visible_memories(Processor p) const
      { return visible_memories_from_procs.find(p)->second; }

      // Return the set of memories visible from a memory
      const std::set<Memory>&    get_visible_memories(Memory m) const
      { return visible_memories_from_memory.find(m)->second; }

      // Return the set of processors which can all see a given memory
      const std::set<Processor>& get_shared_processors(Memory m) const
      { return visible_procs_from_memory.find(m)->second; }

      Processor::Kind get_processor_kind(Processor p) const;
      size_t get_memory_size(const Memory m) const;

      //void add_processor(Processor p) { procs.insert(p); }
      static Machine* get_machine(void);

      struct ProcessorMemoryAffinity {
	Processor p;
	Memory m;
	unsigned bandwidth; // TODO: consider splitting read vs. write?
	unsigned latency;
      };

      struct MemoryMemoryAffinity {
	Memory m1, m2;
	unsigned bandwidth;
	unsigned latency;
      };

      int get_proc_mem_affinity(std::vector<ProcessorMemoryAffinity>& result,
				Processor restrict_proc = Processor::NO_PROC,
				Memory restrict_memory = Memory::NO_MEMORY);

      int get_mem_mem_affinity(std::vector<MemoryMemoryAffinity>& result,
			       Memory restrict_mem1 = Memory::NO_MEMORY,
			       Memory restrict_mem2 = Memory::NO_MEMORY);

    protected:
      std::set<Processor> procs;
      std::set<Memory> memories;
      std::vector<ProcessorMemoryAffinity> proc_mem_affinities;
      std::vector<MemoryMemoryAffinity> mem_mem_affinities;
      std::map<Processor,std::set<Memory> > visible_memories_from_procs;
      std::map<Memory,std::set<Memory> > visible_memories_from_memory;
      std::map<Memory,std::set<Processor> > visible_procs_from_memory;

    public:
      struct NodeAnnounceData;

      void parse_node_announce_data(const void *args, size_t arglen,
				    const NodeAnnounceData& annc_data,
				    bool remote);
    };

  }; // namespace LowLevel
}; // namespace RegionRuntime

#endif
