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
      typedef unsigned long long ID;
      ID id;
      bool operator<(const Event& rhs) const { return id < rhs.id; }
      bool operator==(const Event& rhs) const { return id == rhs.id; }

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
      typedef unsigned ID;
      ID id;
      bool operator<(const Lock& rhs) const { return id < rhs.id; }
      bool operator==(const Lock& rhs) const { return id == rhs.id; }

      // requests ownership (either exclusive or shared) of the lock with a 
      //   specified mode - returns an event that will trigger when the lock
      //   is granted
      Event lock(unsigned mode = 0, bool exclusive = true, Event wait_on = Event::NO_EVENT) const;
      // releases a held lock - release can be deferred until an event triggers
      void unlock(Event wait_on = Event::NO_EVENT) const;

      bool exists(void) const;

      // Create a new lock, destroy an existing lock
      static Lock create_lock(void);
      void destroy_lock();
  };

    class Processor {
    public:
      typedef unsigned ID;
      ID id;
      bool operator<(const Processor& rhs) const { return id < rhs.id; }
      bool operator==(const Processor& rhs) const { return id == rhs.id; }
      bool exists(void) const;
      void register_scheduler(void (*scheduler)(Processor));

      typedef unsigned TaskFuncID;
      typedef void (*TaskFuncPtr)(const void *args, size_t arglen, Processor proc);
      typedef std::map<TaskFuncID, TaskFuncPtr> TaskIDTable;

      Event spawn(TaskFuncID func_id, const void *args, size_t arglen,
		  Event wait_on = Event::NO_EVENT) const;
    };

    class Memory {
    public:
      typedef unsigned ID;
      ID id;
      bool operator<(const Memory &rhs) const { return id < rhs.id; }
      bool operator==(const Memory &rhs) const { return id == rhs.id; }
      bool exists(void) const;
    };

    class ElementMask {
    public:
      ElementMask(int num_elements, int first_element = 0);
      ElementMask(const ElementMask &copy_from, int num_elements = -1, int first_element = 0);

      void enable(int first_element, int count = 1);
      void disable(int first_element, int count = 1);
      
      // is_set?
      // union/intersect/subtract?

      int first_enabled(void) const;
      int last_enabled(void) const;

      size_t raw_size(void) const;
      const void *get_raw(void) const;
      void set_raw(const void *data);

    protected:
      void *raw_data;
    };

    class RegionMetaDataUntyped {
    public:
      typedef unsigned ID;
      ID id;
      bool operator<(const RegionMetaDataUntyped &rhs) const { return id < rhs.id; }
      bool operator==(const RegionMetaDataUntyped &rhs) const { return id == rhs.id; }

      static RegionMetaDataUntyped create_region_untyped(Memory memory, size_t num_elmts, size_t elmt_size);
      RegionAllocatorUntyped create_allocator_untyped(Memory memory) const;
      RegionInstanceUntyped create_instance_untyped(Memory memory) const;
      void destroy_region_untyped() const;
      void destroy_allocator_untyped(RegionAllocatorUntyped allocator) const;
      void destroy_instance_untyped(RegionInstanceUntyped instance) const;

      // get the lock that covers this metadata
      Lock get_lock(void) const;
      bool exists(void) const;

      // it's ok to call these without holding the lock if you don't mind
      //  stale data - data will be up to date if you hold the lock
      RegionAllocatorUntyped get_master_allocator_untyped(void);
      RegionInstanceUntyped get_master_instance_untyped(void);

      // don't call these unless you hold an exclusive lock on the metadata
      void set_master_allocator_untyped(RegionAllocatorUntyped allocator);
      void set_master_instance_untyped(RegionInstanceUntyped instance);
    };

    class RegionAllocatorUntyped {
    public:
      typedef unsigned ID;
      ID id;
      bool operator<(const RegionAllocatorUntyped &rhs) const { return id < rhs.id; }
      bool operator==(const RegionAllocatorUntyped &rhs) const { return id == rhs.id; }

      // get the lock that covers this allocator
      Lock get_lock(void);

      bool exists(void) const;

    protected:
      // can't have virtual methods here, so we're returning function pointers
      typedef unsigned (*AllocFuncPtr)(RegionAllocatorUntyped region, size_t num_elmts);
      typedef void (*FreeFuncPtr)(RegionAllocatorUntyped region, unsigned ptr);

      AllocFuncPtr alloc_fn_untyped(void) const;
      FreeFuncPtr free_fn_untyped(void) const;
    };

    class RegionInstanceUntyped {
    public:
      typedef unsigned ID;
      ID id;
      bool operator<(const RegionInstanceUntyped &rhs) const { return id < rhs.id; }
      bool operator==(const RegionInstanceUntyped &rhs) const { return id == rhs.id; }

      // get the lock that covers this instance
      Lock get_lock(void);

      bool exists(void) const;

      // if non-null, the base of an "array" that can be dereferenced
      void *direct_access_base;

      Event copy_to(RegionInstanceUntyped target, Event wait_on = Event::NO_EVENT)
      { return copy_fn_untyped()(*this, target, wait_on); }
      Event copy_to(RegionInstanceUntyped target, const ElementMask &mask, Event wait_on = Event::NO_EVENT)
      { return copy_fn_untyped()(*this, target, wait_on); }

    protected:
      // can't have virtual methods here, so we're returning function pointers
      typedef const void *(*ReadFuncPtr)(RegionInstanceUntyped region, unsigned ptr);
      typedef void (*WriteFuncPtr)(RegionInstanceUntyped region, unsigned ptr, const void *src);

      ReadFuncPtr read_fn_untyped(void);
      WriteFuncPtr write_fn_untyped(void);
      //UntypedFuncPtr reduce_fn_untyped(void);

      // The copy operation
      typedef Event (*CopyFuncPtr)(RegionInstanceUntyped source, RegionInstanceUntyped target, Event wait_on);

      CopyFuncPtr copy_fn_untyped(void);
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
	ptr_t<T> ptr = { alloc_fn_untyped()(*this, 1) };
	return ptr; 
      }
      void free(ptr_t<T> ptr) { free_fn_untyped()(*this, ptr.value); }
    };

    template <class T>
    class RegionInstance : public RegionInstanceUntyped {
    public:
      // operator to re-introduce element type - make sure you're right!
      explicit RegionInstance(const RegionInstanceUntyped& copy_from)
	: RegionInstanceUntyped(copy_from) {}

      // note the level of indirection here - needed because the base class
      //  can't be virtual
#if 0
      typedef T (*ReadFuncPtr)(ptr_t<T> ptr);
      typedef void (*WriteFuncPtr)(ptr_t<T> ptr, T newval);
      typedef void (*ReduceFuncPtr)(ptr_t<T> ptr, T (*reduce_op)(T, T), T newval);

      ReadFuncPtr read_fn(void) { return (ReadFuncPtr)(read_fn_untyped()); }
      WriteFuncPtr write_fn(void) { return (WriteFuncPtr)(write_fn_untyped()); }
      ReduceFuncPtr reduce_fn(void) { return (ReduceFuncPtr)(reduce_fn_untyped()); }
#endif

      T *make_direct_ptr(void *direct_access_base, ptr_t<T> ptr)
      {
	return ((T*)direct_access_base)+ptr.value;
      }

      T read(ptr_t<T> ptr)
      {
#ifndef FORCE_DIRECT_ACCESS
	if(!direct_access_base)
	  return *(const T*)(read_fn_untyped()(*this, ptr.value));
	else
#endif
	  return *make_direct_ptr(direct_access_base, ptr);
      }

      void write(ptr_t<T> ptr, T newval)
      {
#ifndef FORCE_DIRECT_ACCESS
	if(!direct_access_base)
	  (write_fn_untyped())(*this, ptr.value, (const void *)&newval);
	else
#endif
	  *make_direct_ptr(direct_access_base, ptr) = newval;
      }

#if 1
#else
	T read(ptr_t<T> ptr) { return *((T*)(read_untyped(ptr.value))); }	
	void write(ptr_t<T> ptr, T newval) { write_untyped(ptr.value,((void*)&newval)); }

#endif
      Event copy_to(RegionInstance<T> target, Event wait_on = Event::NO_EVENT)
      { return copy_fn_untyped()(*this, target, wait_on); }

      Event copy_to(RegionInstance<T> target, const ElementMask& mask, Event wait_on = Event::NO_EVENT)
      { return copy_fn_untyped()(*this, target, wait_on); }
    };

    class Machine {
    public:
      Machine(int *argc, char ***argv,
	      const Processor::TaskIDTable &task_table,
	      bool cps_style = false, Processor::TaskFuncID init_id = 0);
      ~Machine(void);

      void run(Processor::TaskFuncID task_id = 0);

    public:
      // Different Processor types
      enum ProcessorKind {
	TOC_PROC, // Throughput core
	LOC_PROC, // Latency core
      };

    public:
      const std::set<Memory>&    get_all_memories(void) const { return memories; }
      const std::set<Processor>& get_all_processors(void) const { return procs; }
      // Return the set of memories visible from a processor
      const std::set<Memory>&    get_visible_memories(const Processor p);
      // Return the set of memories visible from a memory
      const std::set<Memory>&    get_visible_memories(const Memory m);
      // Return the set of processors which can all see a given memory
      const std::set<Processor>& get_shared_processors(const Memory m);

      Processor     get_local_processor() const;
      ProcessorKind get_processor_kind(Processor p) const;
      size_t        get_memory_size(const Memory m) const;

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
