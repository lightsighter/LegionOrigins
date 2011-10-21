#ifndef RUNTIME_LOWLEVEL_H
#define RUNTIME_LOWLEVEL_H

#include <string>
#include <set>
#include <map>

#include "common.h"

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

      class Impl;
      Impl *impl(void) const;

      static const Processor NO_PROC;

      bool exists(void) const { return id != 0; }

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
		  Event wait_on = Event::NO_EVENT) const;
    };

    class Memory {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const Memory &rhs) const { return id < rhs.id; }
      bool operator==(const Memory &rhs) const { return id == rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const Memory NO_MEMORY;

      bool exists(void) const;
    };

    class ElementMask {
    public:
      ElementMask(void);
      ElementMask(int num_elements, int first_element = 0);
      ElementMask(const ElementMask &copy_from, int num_elements = -1, int first_element = 0);

      void init(int _first_element, int _num_elements, Memory _memory, int _offset);

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
      int offset;
      void *raw_data;
    };

    class RegionMetaDataUntyped {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const RegionMetaDataUntyped &rhs) const { return id < rhs.id; }
      bool operator==(const RegionMetaDataUntyped &rhs) const { return id == rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const RegionMetaDataUntyped NO_REGION;

      bool exists(void) const { return id != 0; }

      static RegionMetaDataUntyped create_region_untyped(Memory memory, size_t num_elmts, size_t elmt_size);
      static RegionMetaDataUntyped create_region_untyped(Memory memory, RegionMetaDataUntyped parent, const ElementMask &mask);
      RegionAllocatorUntyped create_allocator_untyped(Memory memory) const;
      RegionInstanceUntyped create_instance_untyped(Memory memory) const;
      void destroy_region_untyped(void) const;
      void destroy_allocator_untyped(RegionAllocatorUntyped allocator) const;
      void destroy_instance_untyped(RegionInstanceUntyped instance) const;

      // get the lock that covers this metadata
      Lock get_lock(void) const;

      // it's ok to call these without holding the lock if you don't mind
      //  stale data - data will be up to date if you hold the lock
      RegionAllocatorUntyped get_master_allocator_untyped(void);
      RegionInstanceUntyped get_master_instance_untyped(void);

      // don't call these unless you hold an exclusive lock on the metadata
      void set_master_allocator_untyped(RegionAllocatorUntyped allocator);
      void set_master_instance_untyped(RegionInstanceUntyped instance);

      const ElementMask &get_valid_mask(void);
    };

    class RegionAllocatorUntyped {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const RegionAllocatorUntyped &rhs) const { return id < rhs.id; }
      bool operator==(const RegionAllocatorUntyped &rhs) const { return id == rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const RegionAllocatorUntyped NO_ALLOC;

      bool exists(void) const { return id != 0; }

      // get the lock that covers this allocator
      Lock get_lock(void);

    protected:
      unsigned alloc_untyped(unsigned count = 1) const;
      void free_untyped(unsigned ptr, unsigned count = 1) const;
    };

    enum AccessorType { AccessorGeneric, AccessorArray };

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
      { T val; get_untyped(ptr.value, &val, sizeof(val)); return val; }

      template <class T>
      void write(ptr_t<T> ptr, T newval) const
      { put_untyped(ptr.value, &newval, sizeof(newval)); }

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

      template <class T, class REDOP>
      void reduce(ptr_t<T> ptr, T newval) const { REDOP::reduce(((T*)array_base)[ptr.value], newval); }
    };

    template <class ET, AccessorType AT = AccessorGeneric>
    class RegionInstanceAccessor {
    public:
      RegionInstanceAccessor(const RegionInstanceAccessorUntyped<AT> &_ria) : ria(_ria) {}

      RegionInstanceAccessorUntyped<AT> ria;

      ET read(ptr_t<ET> ptr) const { return ria.read(ptr); }
      void write(ptr_t<ET> ptr, ET newval) const { ria.write(ptr, newval); }

      template <class REDOP>
      void reduce(ptr_t<ET> ptr, ET newval) const { ria.reduce<REDOP>(ptr, newval); }

      template <AccessorType AT2>
      bool can_convert(void) const { return ria.can_convert<AT2>(); }

      template <AccessorType AT2>
      RegionInstanceAccessor<ET,AT2> convert(void) const
      { return RegionInstanceAccessor<ET,AT2>(ria.convert<AT2>()); }
    };

    class RegionInstanceUntyped {
    public:
      typedef unsigned id_t;
      id_t id;
      bool operator<(const RegionInstanceUntyped &rhs) const { return id < rhs.id; }
      bool operator==(const RegionInstanceUntyped &rhs) const { return id == rhs.id; }

      class Impl;
      Impl *impl(void) const;

      static const RegionInstanceUntyped NO_INST;

      bool exists(void) const { return id != 0; }

      // get the lock that covers this instance
      Lock get_lock(void);

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

      static RegionMetaData<T> create_region(Memory memory, size_t _num_elems) {
	return RegionMetaData<T>(create_region_untyped(memory,_num_elems,sizeof(T)));
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

      // it's ok to call these without holding the lock if you don't mind
      //  stale data - data will be up to date if you hold the lock
      RegionAllocator<T> get_master_allocator(void) {
	return RegionAllocator<T>(get_master_allocator_untyped());
      }

      RegionInstance<T> get_master_instance(void) {
	return RegionInstance<T>(get_master_instance_untyped());
      }

      // don't call these unless you hold an exclusive lock on the metadata
      void set_master_allocator(RegionAllocator<T> allocator) {
	set_master_allocator_untyped(allocator);
      }

      void set_master_instance(RegionInstance<T> instance) {
	set_master_instance_untyped(instance);
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
      // operator to re-introduce element type - make sure you're right!
      explicit RegionInstance(const RegionInstanceUntyped& copy_from)
	: RegionInstanceUntyped(copy_from) {}

      // the instance doesn't have read/write/reduce methods of its own -
      //  instead, we can hand out an "accessor" object that has those methods
      //  this lets us specialize for the just-an-array-dereference case
      const RegionInstanceAccessor<T,AccessorGeneric> get_accessor(void)
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

    protected:
      std::set<Processor> procs;
      std::set<Memory> memories;
      std::map<Processor,std::set<Memory> > visible_memories_from_procs;
      std::map<Memory,std::set<Memory> > visible_memories_from_memory;
      std::map<Memory,std::set<Processor> > visible_procs_from_memory;
    };

  }; // namespace LowLevel
}; // namespace RegionRuntime

#endif
